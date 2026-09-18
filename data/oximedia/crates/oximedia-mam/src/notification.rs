//! Email, Slack, and Microsoft Teams notifications for asset and workflow events.
//!
//! This module provides:
//! - Typed notification events covering the full asset/workflow lifecycle
//! - Channel configuration: email, Slack, Teams (all pure-Rust, no native deps)
//! - A `NotificationDispatcher` that routes events to subscribed channels
//! - Per-user subscription preferences with event-type filtering
//! - Digest batching (aggregate N events before sending)
//! - Delivery receipts and retry tracking

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Notification event types
// ---------------------------------------------------------------------------

/// Domain events that can trigger notifications.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NotificationEvent {
    // Asset lifecycle
    /// An asset was ingested and is ready.
    AssetIngested,
    /// An asset's metadata was updated.
    AssetUpdated,
    /// An asset was deleted (or moved to trash).
    AssetDeleted,
    /// An asset's status changed.
    AssetStatusChanged,
    /// An asset was shared with a user/group.
    AssetShared,
    /// An asset is expiring soon (approaching retention deadline).
    AssetExpiringSoon,

    // Workflow
    /// A workflow was started.
    WorkflowStarted,
    /// A workflow step was completed.
    WorkflowStepCompleted,
    /// A workflow was approved.
    WorkflowApproved,
    /// A workflow was rejected.
    WorkflowRejected,
    /// A workflow task was assigned to the user.
    WorkflowTaskAssigned,

    // Review / collaboration
    /// A comment was added to an asset the user follows.
    CommentAdded,
    /// A review marker was added.
    ReviewMarkerAdded,
    /// A blocking marker was resolved.
    BlockerResolved,
    /// The user was mentioned in a comment.
    MentionReceived,

    // Storage / archive
    /// A storage tier changed (e.g. cold → glacier).
    TierChanged,
    /// An archive retrieval completed.
    RetrievalCompleted,

    // Access control
    /// An access request was submitted.
    AccessRequested,
    /// An access request was approved.
    AccessApproved,
    /// An access request was denied.
    AccessDenied,

    // System / batch
    /// A batch ingest job completed.
    BatchIngestCompleted,
    /// A scheduled report was generated.
    ReportGenerated,

    /// Custom event type for extensibility.
    Custom(String),
}

impl NotificationEvent {
    /// Human-readable display name for the event.
    #[must_use]
    pub fn display_name(&self) -> &str {
        match self {
            Self::AssetIngested => "Asset Ingested",
            Self::AssetUpdated => "Asset Updated",
            Self::AssetDeleted => "Asset Deleted",
            Self::AssetStatusChanged => "Asset Status Changed",
            Self::AssetShared => "Asset Shared",
            Self::AssetExpiringSoon => "Asset Expiring Soon",
            Self::WorkflowStarted => "Workflow Started",
            Self::WorkflowStepCompleted => "Workflow Step Completed",
            Self::WorkflowApproved => "Workflow Approved",
            Self::WorkflowRejected => "Workflow Rejected",
            Self::WorkflowTaskAssigned => "Workflow Task Assigned",
            Self::CommentAdded => "Comment Added",
            Self::ReviewMarkerAdded => "Review Marker Added",
            Self::BlockerResolved => "Blocker Resolved",
            Self::MentionReceived => "Mention Received",
            Self::TierChanged => "Storage Tier Changed",
            Self::RetrievalCompleted => "Archive Retrieval Completed",
            Self::AccessRequested => "Access Requested",
            Self::AccessApproved => "Access Approved",
            Self::AccessDenied => "Access Denied",
            Self::BatchIngestCompleted => "Batch Ingest Completed",
            Self::ReportGenerated => "Report Generated",
            Self::Custom(s) => s.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// Notification payload
// ---------------------------------------------------------------------------

/// A notification message to be delivered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    /// Unique notification id.
    pub id: Uuid,
    /// Event type that triggered this notification.
    pub event: NotificationEvent,
    /// Human-readable title/subject.
    pub title: String,
    /// Notification body (plain text).
    pub body: String,
    /// Optional rich body (HTML for email, markdown for Slack/Teams).
    pub rich_body: Option<String>,
    /// Optional deep-link URL into the MAM application.
    pub action_url: Option<String>,
    /// Structured metadata for templates (arbitrary key-value pairs).
    pub metadata: HashMap<String, serde_json::Value>,
    /// Target recipient user ids.
    pub recipients: Vec<Uuid>,
    /// When this notification was created.
    pub created_at: DateTime<Utc>,
}

impl Notification {
    /// Create a new notification.
    #[must_use]
    pub fn new(
        event: NotificationEvent,
        title: impl Into<String>,
        body: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            event,
            title: title.into(),
            body: body.into(),
            rich_body: None,
            action_url: None,
            metadata: HashMap::new(),
            recipients: Vec::new(),
            created_at: Utc::now(),
        }
    }

    /// Builder: add a recipient.
    #[must_use]
    pub fn to_recipient(mut self, user_id: Uuid) -> Self {
        self.recipients.push(user_id);
        self
    }

    /// Builder: add multiple recipients.
    #[must_use]
    pub fn to_recipients(mut self, user_ids: impl IntoIterator<Item = Uuid>) -> Self {
        self.recipients.extend(user_ids);
        self
    }

    /// Builder: set rich body.
    #[must_use]
    pub fn with_rich_body(mut self, body: impl Into<String>) -> Self {
        self.rich_body = Some(body.into());
        self
    }

    /// Builder: set action URL.
    #[must_use]
    pub fn with_action_url(mut self, url: impl Into<String>) -> Self {
        self.action_url = Some(url.into());
        self
    }

    /// Builder: add metadata entry.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

// ---------------------------------------------------------------------------
// Delivery channels
// ---------------------------------------------------------------------------

/// Configuration for an email notification channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailChannelConfig {
    /// SMTP host.
    pub smtp_host: String,
    /// SMTP port.
    pub smtp_port: u16,
    /// Sender address.
    pub from_address: String,
    /// Sender display name.
    pub from_name: String,
    /// Whether to use TLS.
    pub use_tls: bool,
}

impl Default for EmailChannelConfig {
    fn default() -> Self {
        Self {
            smtp_host: "localhost".to_string(),
            smtp_port: 587,
            from_address: "mam@oximedia.io".to_string(),
            from_name: "OxiMedia MAM".to_string(),
            use_tls: true,
        }
    }
}

/// Configuration for a Slack notification channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackChannelConfig {
    /// Incoming webhook URL.
    pub webhook_url: String,
    /// Default channel name (e.g. `#mam-alerts`).
    pub default_channel: String,
    /// Bot display name.
    pub bot_name: String,
    /// Bot emoji icon.
    pub icon_emoji: String,
}

/// Configuration for a Microsoft Teams notification channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamsChannelConfig {
    /// Incoming webhook URL for the Teams channel.
    pub webhook_url: String,
    /// Activity title prefix.
    pub activity_title_prefix: String,
    /// Accent colour (hex, e.g. `"0076D7"` for Teams blue).
    pub theme_colour: String,
}

impl Default for TeamsChannelConfig {
    fn default() -> Self {
        Self {
            webhook_url: String::new(),
            activity_title_prefix: "OxiMedia MAM".to_string(),
            theme_colour: "0076D7".to_string(),
        }
    }
}

/// A delivery channel variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelConfig {
    Email(EmailChannelConfig),
    Slack(SlackChannelConfig),
    Teams(TeamsChannelConfig),
}

impl ChannelConfig {
    /// Channel type label.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Email(_) => "email",
            Self::Slack(_) => "slack",
            Self::Teams(_) => "teams",
        }
    }
}

// ---------------------------------------------------------------------------
// Delivery record
// ---------------------------------------------------------------------------

/// Status of a notification delivery attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryStatus {
    /// Waiting to be sent.
    Pending,
    /// Successfully delivered.
    Delivered,
    /// Delivery failed; retry may occur.
    Failed,
    /// Maximum retries exceeded; giving up.
    Abandoned,
}

impl DeliveryStatus {
    /// Returns `true` if the delivery was successful.
    #[must_use]
    pub const fn is_delivered(&self) -> bool {
        matches!(self, Self::Delivered)
    }
}

/// Record of a single delivery attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryRecord {
    pub id: Uuid,
    pub notification_id: Uuid,
    pub channel: String,
    pub recipient_id: Uuid,
    pub status: DeliveryStatus,
    pub attempt_count: u32,
    pub last_attempt_at: Option<DateTime<Utc>>,
    pub delivered_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

impl DeliveryRecord {
    /// Create a new pending delivery record.
    #[must_use]
    pub fn new(notification_id: Uuid, channel: impl Into<String>, recipient_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            notification_id,
            channel: channel.into(),
            recipient_id,
            status: DeliveryStatus::Pending,
            attempt_count: 0,
            last_attempt_at: None,
            delivered_at: None,
            error: None,
        }
    }

    /// Record a successful delivery.
    pub fn mark_delivered(&mut self) {
        self.status = DeliveryStatus::Delivered;
        let now = Utc::now();
        self.last_attempt_at = Some(now);
        self.delivered_at = Some(now);
        self.attempt_count += 1;
    }

    /// Record a failed attempt.
    pub fn mark_failed(&mut self, error: impl Into<String>, max_retries: u32) {
        self.attempt_count += 1;
        self.last_attempt_at = Some(Utc::now());
        self.error = Some(error.into());
        if self.attempt_count >= max_retries {
            self.status = DeliveryStatus::Abandoned;
        } else {
            self.status = DeliveryStatus::Failed;
        }
    }
}

// ---------------------------------------------------------------------------
// Subscription preferences
// ---------------------------------------------------------------------------

/// Per-user notification subscription preferences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSubscription {
    /// User this subscription belongs to.
    pub user_id: Uuid,
    /// Events the user wants to receive.
    pub subscribed_events: Vec<NotificationEvent>,
    /// Channels enabled for this user (channel kind → enabled).
    pub enabled_channels: HashMap<String, bool>,
    /// If `true`, batch events into digests rather than sending immediately.
    pub digest_mode: bool,
    /// Maximum number of events to batch before flushing (only for digest_mode).
    pub digest_batch_size: usize,
    /// Email address (if email channel is enabled).
    pub email_address: Option<String>,
    /// Slack user id or DM webhook (if Slack is enabled).
    pub slack_target: Option<String>,
}

impl NotificationSubscription {
    /// Create a new subscription with default settings (all events, all channels).
    #[must_use]
    pub fn new(user_id: Uuid) -> Self {
        let enabled_channels = [
            ("email".to_string(), true),
            ("slack".to_string(), false),
            ("teams".to_string(), false),
        ]
        .into_iter()
        .collect();

        Self {
            user_id,
            subscribed_events: default_subscribed_events(),
            enabled_channels,
            digest_mode: false,
            digest_batch_size: 10,
            email_address: None,
            slack_target: None,
        }
    }

    /// Returns `true` if the user is subscribed to the given event.
    #[must_use]
    pub fn is_subscribed(&self, event: &NotificationEvent) -> bool {
        self.subscribed_events.contains(event)
    }

    /// Returns `true` if the given channel is enabled.
    #[must_use]
    pub fn is_channel_enabled(&self, channel: &str) -> bool {
        self.enabled_channels.get(channel).copied().unwrap_or(false)
    }

    /// Subscribe to an additional event.
    pub fn subscribe(&mut self, event: NotificationEvent) {
        if !self.subscribed_events.contains(&event) {
            self.subscribed_events.push(event);
        }
    }

    /// Unsubscribe from an event.
    pub fn unsubscribe(&mut self, event: &NotificationEvent) {
        self.subscribed_events.retain(|e| e != event);
    }
}

fn default_subscribed_events() -> Vec<NotificationEvent> {
    vec![
        NotificationEvent::AssetIngested,
        NotificationEvent::WorkflowTaskAssigned,
        NotificationEvent::WorkflowApproved,
        NotificationEvent::WorkflowRejected,
        NotificationEvent::MentionReceived,
        NotificationEvent::AccessRequested,
        NotificationEvent::BatchIngestCompleted,
    ]
}

// ---------------------------------------------------------------------------
// Digest buffer
// ---------------------------------------------------------------------------

/// Accumulates notifications for a user before flushing.
#[derive(Debug)]
pub struct DigestBuffer {
    user_id: Uuid,
    batch_size: usize,
    pending: Vec<Notification>,
}

impl DigestBuffer {
    /// Create a new digest buffer.
    #[must_use]
    pub fn new(user_id: Uuid, batch_size: usize) -> Self {
        Self {
            user_id,
            batch_size,
            pending: Vec::new(),
        }
    }

    /// Add a notification to the buffer.
    ///
    /// Returns `Some(Vec<Notification>)` if the batch is ready to flush.
    pub fn push(&mut self, notification: Notification) -> Option<Vec<Notification>> {
        self.pending.push(notification);
        if self.pending.len() >= self.batch_size {
            Some(self.flush())
        } else {
            None
        }
    }

    /// Flush all pending notifications.
    #[must_use]
    pub fn flush(&mut self) -> Vec<Notification> {
        std::mem::take(&mut self.pending)
    }

    /// Number of notifications waiting.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// The user this buffer belongs to.
    #[must_use]
    pub fn user_id(&self) -> Uuid {
        self.user_id
    }
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Routes notifications to subscribed users across multiple channels.
///
/// The dispatcher is intentionally decoupled from actual HTTP/SMTP sending so
/// it can be used in tests without network I/O.  Transport is modelled via the
/// `NotificationTransport` trait.
pub struct NotificationDispatcher {
    /// Registered channels (channel kind → config).
    channels: HashMap<String, ChannelConfig>,
    /// Per-user subscriptions.
    subscriptions: HashMap<Uuid, NotificationSubscription>,
    /// Digest buffers (only populated when digest_mode is enabled).
    digest_buffers: HashMap<Uuid, DigestBuffer>,
    /// All delivery records produced by this dispatcher.
    pub delivery_log: Vec<DeliveryRecord>,
}

impl NotificationDispatcher {
    /// Create a new empty dispatcher.
    #[must_use]
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
            subscriptions: HashMap::new(),
            digest_buffers: HashMap::new(),
            delivery_log: Vec::new(),
        }
    }

    /// Register a channel.
    pub fn register_channel(&mut self, config: ChannelConfig) {
        self.channels.insert(config.kind().to_string(), config);
    }

    /// Add or update a user subscription.
    pub fn set_subscription(&mut self, sub: NotificationSubscription) {
        let uid = sub.user_id;
        let batch = sub.digest_batch_size;
        let digest = sub.digest_mode;
        self.subscriptions.insert(uid, sub);
        if digest {
            self.digest_buffers
                .entry(uid)
                .or_insert_with(|| DigestBuffer::new(uid, batch));
        }
    }

    /// Get a subscription by user id.
    #[must_use]
    pub fn subscription(&self, user_id: Uuid) -> Option<&NotificationSubscription> {
        self.subscriptions.get(&user_id)
    }

    /// Dispatch a notification to all subscribed recipients.
    ///
    /// Returns the number of delivery records created.
    pub fn dispatch(&mut self, notification: &Notification) -> usize {
        let mut created = 0;
        let recipients: Vec<Uuid> = notification.recipients.clone();

        for uid in recipients {
            let sub = match self.subscriptions.get(&uid) {
                Some(s) => s,
                None => continue,
            };

            if !sub.is_subscribed(&notification.event) {
                continue;
            }

            // Collect channels that are active for this user
            let active_channels: Vec<String> = self
                .channels
                .keys()
                .filter(|k| sub.is_channel_enabled(k))
                .cloned()
                .collect();

            if sub.digest_mode {
                // Buffer and potentially flush
                if let Some(buffer) = self.digest_buffers.get_mut(&uid) {
                    let flushed = buffer.push(notification.clone());
                    if let Some(batch) = flushed {
                        for n in &batch {
                            for ch in &active_channels {
                                let record = DeliveryRecord::new(n.id, ch.clone(), uid);
                                self.delivery_log.push(record);
                                created += 1;
                            }
                        }
                    }
                }
            } else {
                for ch in &active_channels {
                    let record = DeliveryRecord::new(notification.id, ch.clone(), uid);
                    self.delivery_log.push(record);
                    created += 1;
                }
            }
        }

        created
    }

    /// Force-flush all digest buffers and create delivery records.
    pub fn flush_all_digests(&mut self) -> usize {
        let user_ids: Vec<Uuid> = self.digest_buffers.keys().copied().collect();
        let mut created = 0;

        for uid in user_ids {
            let active_channels: Vec<String> = if let Some(sub) = self.subscriptions.get(&uid) {
                self.channels
                    .keys()
                    .filter(|k| sub.is_channel_enabled(k))
                    .cloned()
                    .collect()
            } else {
                vec![]
            };

            if let Some(buffer) = self.digest_buffers.get_mut(&uid) {
                let batch = buffer.flush();
                for n in &batch {
                    for ch in &active_channels {
                        let record = DeliveryRecord::new(n.id, ch.clone(), uid);
                        self.delivery_log.push(record);
                        created += 1;
                    }
                }
            }
        }

        created
    }

    /// Build a Slack JSON payload for the given notification.
    ///
    /// Produces a JSON `{"text": ..., "blocks": [...]}` body suitable for
    /// the Slack Incoming Webhooks API.
    #[must_use]
    pub fn build_slack_payload(&self, notification: &Notification) -> String {
        let text = format!("*{}*\n{}", notification.title, notification.body);
        let mut block = serde_json::json!({
            "type": "section",
            "text": { "type": "mrkdwn", "text": text }
        });

        if let Some(url) = &notification.action_url {
            block["accessory"] = serde_json::json!({
                "type": "button",
                "text": { "type": "plain_text", "text": "Open" },
                "url": url
            });
        }

        serde_json::json!({
            "text": format!("{} — {}", notification.event.display_name(), notification.title),
            "blocks": [block]
        })
        .to_string()
    }

    /// Build a Microsoft Teams Adaptive Card JSON body for the given notification.
    #[must_use]
    pub fn build_teams_payload(
        &self,
        notification: &Notification,
        config: &TeamsChannelConfig,
    ) -> String {
        serde_json::json!({
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "themeColor": config.theme_colour,
            "summary": notification.title,
            "sections": [{
                "activityTitle": format!("{}: {}", config.activity_title_prefix, notification.title),
                "activityText": notification.body,
                "facts": notification.metadata.iter().map(|(k, v)| {
                    serde_json::json!({ "name": k, "value": v.to_string() })
                }).collect::<Vec<_>>()
            }]
        })
        .to_string()
    }

    /// Build a plain-text email body for the given notification.
    #[must_use]
    pub fn build_email_body(&self, notification: &Notification) -> String {
        let mut out = format!("{}\n\n{}", notification.title, notification.body);
        if let Some(url) = &notification.action_url {
            out.push_str(&format!("\n\nView: {url}"));
        }
        out
    }

    /// Number of channels registered.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Number of user subscriptions.
    #[must_use]
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }
}

impl Default for NotificationDispatcher {
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

    fn make_dispatcher_with_user(uid: Uuid) -> NotificationDispatcher {
        let mut d = NotificationDispatcher::new();
        d.register_channel(ChannelConfig::Email(EmailChannelConfig::default()));

        let mut sub = NotificationSubscription::new(uid);
        sub.enabled_channels.insert("email".to_string(), true);
        d.set_subscription(sub);
        d
    }

    // --- NotificationEvent ---

    #[test]
    fn test_event_display_names() {
        assert_eq!(
            NotificationEvent::AssetIngested.display_name(),
            "Asset Ingested"
        );
        assert_eq!(
            NotificationEvent::WorkflowApproved.display_name(),
            "Workflow Approved"
        );
        assert_eq!(
            NotificationEvent::Custom("my.event".to_string()).display_name(),
            "my.event"
        );
    }

    // --- Notification builder ---

    #[test]
    fn test_notification_builder() {
        let uid1 = uid();
        let uid2 = uid();
        let n = Notification::new(
            NotificationEvent::AssetIngested,
            "New Asset",
            "An asset arrived",
        )
        .to_recipient(uid1)
        .to_recipient(uid2)
        .with_action_url("https://mam.example.com/assets/123")
        .with_meta("asset_id", serde_json::json!("asset-abc"));

        assert_eq!(n.recipients.len(), 2);
        assert!(n.action_url.is_some());
        assert!(n.metadata.contains_key("asset_id"));
    }

    #[test]
    fn test_notification_rich_body() {
        let n = Notification::new(NotificationEvent::CommentAdded, "New comment", "plain")
            .with_rich_body("<b>rich</b>");
        assert_eq!(n.rich_body.as_deref(), Some("<b>rich</b>"));
    }

    // --- DeliveryRecord ---

    #[test]
    fn test_delivery_record_lifecycle() {
        let mut rec = DeliveryRecord::new(uid(), "email", uid());
        assert_eq!(rec.status, DeliveryStatus::Pending);

        rec.mark_delivered();
        assert!(rec.status.is_delivered());
        assert!(rec.delivered_at.is_some());
        assert_eq!(rec.attempt_count, 1);
    }

    #[test]
    fn test_delivery_record_fail_then_abandon() {
        let mut rec = DeliveryRecord::new(uid(), "email", uid());
        rec.mark_failed("SMTP timeout", 3);
        assert_eq!(rec.status, DeliveryStatus::Failed);
        assert_eq!(rec.attempt_count, 1);

        rec.mark_failed("SMTP timeout", 3);
        assert_eq!(rec.status, DeliveryStatus::Failed);

        rec.mark_failed("SMTP timeout", 3);
        assert_eq!(rec.status, DeliveryStatus::Abandoned);
    }

    // --- NotificationSubscription ---

    #[test]
    fn test_subscription_subscribe_unsubscribe() {
        let mut sub = NotificationSubscription::new(uid());
        sub.subscribe(NotificationEvent::ReportGenerated);
        assert!(sub.is_subscribed(&NotificationEvent::ReportGenerated));

        sub.unsubscribe(&NotificationEvent::ReportGenerated);
        assert!(!sub.is_subscribed(&NotificationEvent::ReportGenerated));
    }

    #[test]
    fn test_subscription_channel_enabled() {
        let mut sub = NotificationSubscription::new(uid());
        assert!(sub.is_channel_enabled("email"));
        assert!(!sub.is_channel_enabled("slack"));

        sub.enabled_channels.insert("slack".to_string(), true);
        assert!(sub.is_channel_enabled("slack"));
    }

    // --- DigestBuffer ---

    #[test]
    fn test_digest_buffer_no_flush_below_threshold() {
        let mut buf = DigestBuffer::new(uid(), 3);
        let n1 = Notification::new(NotificationEvent::AssetIngested, "A1", "b");
        let n2 = Notification::new(NotificationEvent::AssetIngested, "A2", "b");
        assert!(buf.push(n1).is_none());
        assert!(buf.push(n2).is_none());
        assert_eq!(buf.pending_count(), 2);
    }

    #[test]
    fn test_digest_buffer_flush_at_threshold() {
        let mut buf = DigestBuffer::new(uid(), 2);
        let n1 = Notification::new(NotificationEvent::AssetIngested, "A1", "b");
        let n2 = Notification::new(NotificationEvent::AssetIngested, "A2", "b");
        buf.push(n1);
        let flushed = buf.push(n2);
        assert!(flushed.is_some());
        assert_eq!(flushed.expect("flush should succeed").len(), 2);
        assert_eq!(buf.pending_count(), 0);
    }

    #[test]
    fn test_digest_buffer_manual_flush() {
        let mut buf = DigestBuffer::new(uid(), 100);
        for i in 0..5 {
            let n = Notification::new(NotificationEvent::AssetIngested, format!("N{i}"), "b");
            buf.push(n);
        }
        let flushed = buf.flush();
        assert_eq!(flushed.len(), 5);
        assert_eq!(buf.pending_count(), 0);
    }

    // --- NotificationDispatcher ---

    #[test]
    fn test_dispatcher_dispatch_immediate() {
        let user = uid();
        let mut d = make_dispatcher_with_user(user);

        let n = Notification::new(NotificationEvent::AssetIngested, "Ready", "Your file")
            .to_recipient(user);
        let count = d.dispatch(&n);
        assert_eq!(count, 1);
        assert_eq!(d.delivery_log.len(), 1);
        assert_eq!(d.delivery_log[0].recipient_id, user);
        assert_eq!(d.delivery_log[0].channel, "email");
    }

    #[test]
    fn test_dispatcher_event_not_subscribed() {
        let user = uid();
        let mut d = NotificationDispatcher::new();
        d.register_channel(ChannelConfig::Email(EmailChannelConfig::default()));

        let mut sub = NotificationSubscription::new(user);
        // No subscription to ReportGenerated
        sub.subscribed_events.clear();
        d.set_subscription(sub);

        let n = Notification::new(NotificationEvent::ReportGenerated, "Report", "Ready")
            .to_recipient(user);
        let count = d.dispatch(&n);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_dispatcher_digest_mode() {
        let user = uid();
        let mut d = NotificationDispatcher::new();
        d.register_channel(ChannelConfig::Email(EmailChannelConfig::default()));

        let mut sub = NotificationSubscription::new(user);
        sub.enabled_channels.insert("email".to_string(), true);
        sub.digest_mode = true;
        sub.digest_batch_size = 3;
        d.set_subscription(sub);

        // Send 2 — not yet flushed
        for _ in 0..2 {
            let n =
                Notification::new(NotificationEvent::AssetIngested, "T", "B").to_recipient(user);
            d.dispatch(&n);
        }
        assert_eq!(d.delivery_log.len(), 0);

        // Send 3rd — flush triggered
        let n = Notification::new(NotificationEvent::AssetIngested, "T3", "B3").to_recipient(user);
        d.dispatch(&n);
        assert_eq!(d.delivery_log.len(), 3);
    }

    #[test]
    fn test_dispatcher_flush_all_digests() {
        let user = uid();
        let mut d = NotificationDispatcher::new();
        d.register_channel(ChannelConfig::Email(EmailChannelConfig::default()));

        let mut sub = NotificationSubscription::new(user);
        sub.enabled_channels.insert("email".to_string(), true);
        sub.digest_mode = true;
        sub.digest_batch_size = 100;
        d.set_subscription(sub);

        for _ in 0..4 {
            let n =
                Notification::new(NotificationEvent::AssetIngested, "T", "B").to_recipient(user);
            d.dispatch(&n);
        }
        assert_eq!(d.delivery_log.len(), 0);

        let flushed = d.flush_all_digests();
        assert_eq!(flushed, 4);
        assert_eq!(d.delivery_log.len(), 4);
    }

    #[test]
    fn test_build_slack_payload() {
        let d = NotificationDispatcher::new();
        let n = Notification::new(
            NotificationEvent::WorkflowApproved,
            "Approved!",
            "Your asset passed review",
        )
        .with_action_url("https://mam.example.com/assets/42");
        let payload = d.build_slack_payload(&n);
        assert!(payload.contains("Approved!"));
        assert!(payload.contains("Open"));
    }

    #[test]
    fn test_build_teams_payload() {
        let d = NotificationDispatcher::new();
        let cfg = TeamsChannelConfig::default();
        let n = Notification::new(
            NotificationEvent::AssetExpiringSoon,
            "Expiring",
            "3 days left",
        )
        .with_meta("asset", serde_json::json!("abc123"));
        let payload = d.build_teams_payload(&n, &cfg);
        assert!(payload.contains("Expiring"));
        assert!(payload.contains("OxiMedia MAM"));
    }

    #[test]
    fn test_build_email_body() {
        let d = NotificationDispatcher::new();
        let n = Notification::new(
            NotificationEvent::BatchIngestCompleted,
            "Batch Done",
            "All 50 files",
        )
        .with_action_url("https://mam.example.com/batch/99");
        let body = d.build_email_body(&n);
        assert!(body.contains("Batch Done"));
        assert!(body.contains("All 50 files"));
        assert!(body.contains("https://mam.example.com/batch/99"));
    }

    #[test]
    fn test_dispatcher_channel_and_subscription_counts() {
        let mut d = NotificationDispatcher::new();
        d.register_channel(ChannelConfig::Email(EmailChannelConfig::default()));
        d.register_channel(ChannelConfig::Teams(TeamsChannelConfig::default()));
        d.set_subscription(NotificationSubscription::new(uid()));
        d.set_subscription(NotificationSubscription::new(uid()));

        assert_eq!(d.channel_count(), 2);
        assert_eq!(d.subscription_count(), 2);
    }

    #[test]
    fn test_channel_config_kind() {
        assert_eq!(
            ChannelConfig::Email(EmailChannelConfig::default()).kind(),
            "email"
        );
        assert_eq!(
            ChannelConfig::Teams(TeamsChannelConfig::default()).kind(),
            "teams"
        );
    }
}
