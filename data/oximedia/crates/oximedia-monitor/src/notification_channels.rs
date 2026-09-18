//! PagerDuty and OpsGenie alerting channel integrations.
//!
//! Provides pure-Rust implementations of the PagerDuty Events API v2 and the
//! OpsGenie Alerts API for dispatching critical alerts from the monitoring
//! pipeline to on-call teams.
//!
//! Both integrations serialize alert payloads to JSON and are designed to work
//! alongside the existing Slack/webhook channels in [`crate::alert`].
//!
//! # PagerDuty
//!
//! Uses the [Events API v2](https://developer.pagerduty.com/docs/events-api-v2/overview/).
//! Requires an **integration key** (also called routing key) from the PagerDuty
//! service configuration.
//!
//! # OpsGenie
//!
//! Uses the [Alerts API](https://docs.opsgenie.com/docs/alert-api).
//! Requires an **API key** from the OpsGenie integration settings.

#![allow(dead_code)]

use std::collections::BTreeMap;

// `MonitorError`/`MonitorResult` are only used by the `send()` methods below,
// which perform real HTTP requests via `reqwest` and are therefore gated out
// on `wasm32` (no blocking-friendly HTTP client there).
#[cfg(not(target_arch = "wasm32"))]
use crate::error::{MonitorError, MonitorResult};

// ---------------------------------------------------------------------------
// Severity mapping
// ---------------------------------------------------------------------------

/// Unified alert severity for notification channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotifySeverity {
    /// Informational only.
    Info,
    /// Warning — attention needed.
    Warning,
    /// Critical — immediate action required.
    Critical,
}

impl NotifySeverity {
    /// Map to PagerDuty `severity` field.
    #[must_use]
    pub fn pagerduty_severity(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Critical => "critical",
        }
    }

    /// Map to OpsGenie `priority` field (P1–P5).
    #[must_use]
    pub fn opsgenie_priority(self) -> &'static str {
        match self {
            Self::Info => "P5",
            Self::Warning => "P3",
            Self::Critical => "P1",
        }
    }
}

// ---------------------------------------------------------------------------
// Notification payload
// ---------------------------------------------------------------------------

/// A generic alert notification payload used by both channels.
#[derive(Debug, Clone)]
pub struct NotificationPayload {
    /// Short human-readable summary of the alert.
    pub summary: String,
    /// Alert severity.
    pub severity: NotifySeverity,
    /// Optional source identifier (e.g. host name or service name).
    pub source: Option<String>,
    /// Optional component that is alerting (e.g. `cpu`, `encoding-pipeline`).
    pub component: Option<String>,
    /// Optional longer description.
    pub details: Option<String>,
    /// Arbitrary key-value metadata.
    pub labels: BTreeMap<String, String>,
    /// Unique deduplication key (prevents duplicate pages for the same event).
    pub dedup_key: Option<String>,
}

impl NotificationPayload {
    /// Create a minimal payload.
    #[must_use]
    pub fn new(summary: impl Into<String>, severity: NotifySeverity) -> Self {
        Self {
            summary: summary.into(),
            severity,
            source: None,
            component: None,
            details: None,
            labels: BTreeMap::new(),
            dedup_key: None,
        }
    }

    /// Attach a source identifier.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Attach a component identifier.
    #[must_use]
    pub fn with_component(mut self, component: impl Into<String>) -> Self {
        self.component = Some(component.into());
        self
    }

    /// Attach a detailed description.
    #[must_use]
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Add a label.
    #[must_use]
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Set the deduplication key.
    #[must_use]
    pub fn with_dedup_key(mut self, key: impl Into<String>) -> Self {
        self.dedup_key = Some(key.into());
        self
    }
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

fn json_str(s: &str) -> String {
    format!(
        "\"{}\"",
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    )
}

fn json_field(key: &str, value: &str) -> String {
    format!("{}: {}", json_str(key), json_str(value))
}

fn json_object(fields: &[String]) -> String {
    format!("{{{}}}", fields.join(","))
}

// ---------------------------------------------------------------------------
// PagerDuty
// ---------------------------------------------------------------------------

/// PagerDuty Events API v2 action type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagerDutyAction {
    /// Trigger a new incident.
    Trigger,
    /// Acknowledge an existing incident.
    Acknowledge,
    /// Resolve an existing incident.
    Resolve,
}

impl PagerDutyAction {
    /// The API string value.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Trigger => "trigger",
            Self::Acknowledge => "acknowledge",
            Self::Resolve => "resolve",
        }
    }
}

/// PagerDuty Events API v2 configuration.
#[derive(Debug, Clone)]
pub struct PagerDutyConfig {
    /// Integration (routing) key from the PagerDuty service.
    pub integration_key: String,
    /// API endpoint (default: `https://events.pagerduty.com/v2/enqueue`).
    pub endpoint: String,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
}

impl PagerDutyConfig {
    /// Create a config with the given integration key.
    #[must_use]
    pub fn new(integration_key: impl Into<String>) -> Self {
        Self {
            integration_key: integration_key.into(),
            endpoint: "https://events.pagerduty.com/v2/enqueue".to_string(),
            timeout_ms: 10_000,
        }
    }

    /// Override the API endpoint (useful for testing with a mock).
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Set the request timeout.
    #[must_use]
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }
}

/// PagerDuty notification channel.
#[derive(Debug, Clone)]
pub struct PagerDutyChannel {
    config: PagerDutyConfig,
}

impl PagerDutyChannel {
    /// Create a new channel.
    #[must_use]
    pub fn new(config: PagerDutyConfig) -> Self {
        Self { config }
    }

    /// Serialize a notification payload to PagerDuty Events API v2 JSON.
    ///
    /// `action` controls whether this is a trigger, acknowledge, or resolve.
    #[must_use]
    pub fn serialize(&self, payload: &NotificationPayload, action: PagerDutyAction) -> String {
        let mut details_fields: Vec<String> = Vec::new();
        for (k, v) in &payload.labels {
            details_fields.push(json_field(k, v));
        }
        if let Some(ref d) = payload.details {
            details_fields.push(json_field("description", d));
        }
        let details_obj = json_object(&details_fields);

        let mut pd_payload_fields = vec![
            json_field("summary", &payload.summary),
            json_field("severity", payload.severity.pagerduty_severity()),
        ];
        if let Some(ref src) = payload.source {
            pd_payload_fields.push(json_field("source", src));
        }
        if let Some(ref comp) = payload.component {
            pd_payload_fields.push(json_field("component", comp));
        }
        pd_payload_fields.push(format!("{}: {details_obj}", json_str("custom_details")));
        let pd_payload_obj = json_object(&pd_payload_fields);

        let mut top_fields = vec![
            json_field("routing_key", &self.config.integration_key),
            json_field("event_action", action.as_str()),
            format!("{}: {pd_payload_obj}", json_str("payload")),
        ];
        if let Some(ref key) = payload.dedup_key {
            top_fields.push(json_field("dedup_key", key));
        }

        json_object(&top_fields)
    }

    /// Send a PagerDuty event.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails or the server returns a
    /// non-2xx status.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn send(
        &self,
        payload: &NotificationPayload,
        action: PagerDutyAction,
    ) -> MonitorResult<()> {
        use std::time::Duration;

        let body = self.serialize(payload, action);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(self.config.timeout_ms))
            .build()
            .map_err(MonitorError::Http)?;

        let response = client
            .post(&self.config.endpoint)
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await
            .map_err(MonitorError::Http)?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<unreadable>"));
            return Err(MonitorError::Other(format!(
                "PagerDuty API error: HTTP {status} — {text}"
            )));
        }

        Ok(())
    }

    /// Reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &PagerDutyConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// OpsGenie
// ---------------------------------------------------------------------------

/// OpsGenie Alerts API configuration.
#[derive(Debug, Clone)]
pub struct OpsGenieConfig {
    /// API key from OpsGenie integration settings.
    pub api_key: String,
    /// API endpoint (default: EU API `https://api.eu.opsgenie.com/v2/alerts`).
    pub endpoint: String,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Optional responders (teams or users to notify).
    pub responders: Vec<String>,
    /// Optional tags to add to every alert.
    pub default_tags: Vec<String>,
}

impl OpsGenieConfig {
    /// Create a config with the given API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: "https://api.eu.opsgenie.com/v2/alerts".to_string(),
            timeout_ms: 10_000,
            responders: Vec::new(),
            default_tags: Vec::new(),
        }
    }

    /// Override the API endpoint.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Add a responder (team name).
    #[must_use]
    pub fn with_responder(mut self, responder: impl Into<String>) -> Self {
        self.responders.push(responder.into());
        self
    }

    /// Add a default tag.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.default_tags.push(tag.into());
        self
    }

    /// Set the request timeout.
    #[must_use]
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }
}

/// OpsGenie notification channel.
#[derive(Debug, Clone)]
pub struct OpsGenieChannel {
    config: OpsGenieConfig,
}

impl OpsGenieChannel {
    /// Create a new channel.
    #[must_use]
    pub fn new(config: OpsGenieConfig) -> Self {
        Self { config }
    }

    /// Serialize a notification payload to OpsGenie Alerts API JSON.
    #[must_use]
    pub fn serialize(&self, payload: &NotificationPayload) -> String {
        // Build tags array.
        let mut tags: Vec<String> = self
            .config
            .default_tags
            .iter()
            .map(|t| json_str(t))
            .collect();
        for (k, v) in &payload.labels {
            tags.push(json_str(&format!("{k}:{v}")));
        }
        let tags_array = format!("[{}]", tags.join(","));

        // Build details object.
        let mut details_fields: Vec<String> = Vec::new();
        if let Some(ref d) = payload.details {
            details_fields.push(json_field("description", d));
        }
        if let Some(ref src) = payload.source {
            details_fields.push(json_field("source", src));
        }
        if let Some(ref comp) = payload.component {
            details_fields.push(json_field("component", comp));
        }
        let details_obj = json_object(&details_fields);

        // Build responders array.
        let responders: Vec<String> = self
            .config
            .responders
            .iter()
            .map(|r| json_object(&[json_field("name", r), json_field("type", "team")]))
            .collect();
        let responders_array = format!("[{}]", responders.join(","));

        let mut fields = vec![
            json_field("message", &payload.summary),
            json_field("priority", payload.severity.opsgenie_priority()),
            format!("{}: {tags_array}", json_str("tags")),
            format!("{}: {details_obj}", json_str("details")),
            format!("{}: {responders_array}", json_str("responders")),
        ];
        if let Some(ref key) = payload.dedup_key {
            fields.push(json_field("alias", key));
        }

        json_object(&fields)
    }

    /// Send an alert to OpsGenie.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails or the server returns a
    /// non-2xx status.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn send(&self, payload: &NotificationPayload) -> MonitorResult<()> {
        use std::time::Duration;

        let body = self.serialize(payload);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(self.config.timeout_ms))
            .build()
            .map_err(MonitorError::Http)?;

        let response = client
            .post(&self.config.endpoint)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("GenieKey {}", self.config.api_key))
            .body(body)
            .send()
            .await
            .map_err(MonitorError::Http)?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<unreadable>"));
            return Err(MonitorError::Other(format!(
                "OpsGenie API error: HTTP {status} — {text}"
            )));
        }

        Ok(())
    }

    /// Reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &OpsGenieConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- NotifySeverity --

    #[test]
    fn test_pagerduty_severity_mapping() {
        assert_eq!(NotifySeverity::Info.pagerduty_severity(), "info");
        assert_eq!(NotifySeverity::Warning.pagerduty_severity(), "warning");
        assert_eq!(NotifySeverity::Critical.pagerduty_severity(), "critical");
    }

    #[test]
    fn test_opsgenie_priority_mapping() {
        assert_eq!(NotifySeverity::Info.opsgenie_priority(), "P5");
        assert_eq!(NotifySeverity::Warning.opsgenie_priority(), "P3");
        assert_eq!(NotifySeverity::Critical.opsgenie_priority(), "P1");
    }

    // -- NotificationPayload builder --

    #[test]
    fn test_payload_builder() {
        let p = NotificationPayload::new("High CPU", NotifySeverity::Critical)
            .with_source("server-01")
            .with_component("cpu")
            .with_details("CPU exceeded 90% for 5 minutes")
            .with_label("region", "us-east-1")
            .with_dedup_key("cpu-high-server-01");

        assert_eq!(p.summary, "High CPU");
        assert_eq!(p.severity, NotifySeverity::Critical);
        assert_eq!(p.source.as_deref(), Some("server-01"));
        assert_eq!(p.component.as_deref(), Some("cpu"));
        assert!(p.details.is_some());
        assert_eq!(
            p.labels.get("region").map(String::as_str),
            Some("us-east-1")
        );
        assert_eq!(p.dedup_key.as_deref(), Some("cpu-high-server-01"));
    }

    // -- PagerDutyChannel serialization --

    #[test]
    fn test_pagerduty_serialize_trigger_contains_routing_key() {
        let cfg = PagerDutyConfig::new("test-routing-key-abc123");
        let ch = PagerDutyChannel::new(cfg);
        let payload = NotificationPayload::new("Test alert", NotifySeverity::Warning);
        let json = ch.serialize(&payload, PagerDutyAction::Trigger);
        assert!(
            json.contains("test-routing-key-abc123"),
            "JSON must contain routing key"
        );
    }

    #[test]
    fn test_pagerduty_serialize_trigger_action() {
        let ch = PagerDutyChannel::new(PagerDutyConfig::new("key"));
        let payload = NotificationPayload::new("Alert", NotifySeverity::Critical);
        let json = ch.serialize(&payload, PagerDutyAction::Trigger);
        assert!(json.contains("\"trigger\""));
    }

    #[test]
    fn test_pagerduty_serialize_resolve_action() {
        let ch = PagerDutyChannel::new(PagerDutyConfig::new("key"));
        let payload = NotificationPayload::new("Resolved", NotifySeverity::Info);
        let json = ch.serialize(&payload, PagerDutyAction::Resolve);
        assert!(json.contains("\"resolve\""));
    }

    #[test]
    fn test_pagerduty_serialize_acknowledge_action() {
        let ch = PagerDutyChannel::new(PagerDutyConfig::new("key"));
        let payload = NotificationPayload::new("Ack", NotifySeverity::Warning);
        let json = ch.serialize(&payload, PagerDutyAction::Acknowledge);
        assert!(json.contains("\"acknowledge\""));
    }

    #[test]
    fn test_pagerduty_serialize_severity_in_payload() {
        let ch = PagerDutyChannel::new(PagerDutyConfig::new("key"));
        let payload = NotificationPayload::new("Critical!", NotifySeverity::Critical);
        let json = ch.serialize(&payload, PagerDutyAction::Trigger);
        assert!(json.contains("\"critical\""));
    }

    #[test]
    fn test_pagerduty_serialize_summary_in_payload() {
        let ch = PagerDutyChannel::new(PagerDutyConfig::new("key"));
        let payload = NotificationPayload::new("My Alert Summary", NotifySeverity::Warning);
        let json = ch.serialize(&payload, PagerDutyAction::Trigger);
        assert!(json.contains("My Alert Summary"));
    }

    #[test]
    fn test_pagerduty_serialize_dedup_key() {
        let ch = PagerDutyChannel::new(PagerDutyConfig::new("key"));
        let payload = NotificationPayload::new("Alert", NotifySeverity::Critical)
            .with_dedup_key("unique-incident-id-42");
        let json = ch.serialize(&payload, PagerDutyAction::Trigger);
        assert!(json.contains("unique-incident-id-42"));
    }

    #[test]
    fn test_pagerduty_serialize_source_and_component() {
        let ch = PagerDutyChannel::new(PagerDutyConfig::new("key"));
        let payload = NotificationPayload::new("Alert", NotifySeverity::Warning)
            .with_source("encoder-host-5")
            .with_component("encoding-pipeline");
        let json = ch.serialize(&payload, PagerDutyAction::Trigger);
        assert!(json.contains("encoder-host-5"));
        assert!(json.contains("encoding-pipeline"));
    }

    #[test]
    fn test_pagerduty_serialize_labels_in_custom_details() {
        let ch = PagerDutyChannel::new(PagerDutyConfig::new("key"));
        let payload = NotificationPayload::new("Alert", NotifySeverity::Warning)
            .with_label("region", "eu-west-1");
        let json = ch.serialize(&payload, PagerDutyAction::Trigger);
        assert!(json.contains("eu-west-1"));
    }

    #[test]
    fn test_pagerduty_config_default_endpoint() {
        let cfg = PagerDutyConfig::new("key");
        assert!(cfg.endpoint.contains("pagerduty.com"));
    }

    #[test]
    fn test_pagerduty_config_custom_endpoint() {
        let cfg = PagerDutyConfig::new("key").with_endpoint("http://mock:8080/pagerduty");
        assert_eq!(cfg.endpoint, "http://mock:8080/pagerduty");
    }

    // -- OpsGenieChannel serialization --

    #[test]
    fn test_opsgenie_serialize_contains_message() {
        let cfg = OpsGenieConfig::new("test-api-key");
        let ch = OpsGenieChannel::new(cfg);
        let payload = NotificationPayload::new("Disk near capacity", NotifySeverity::Critical);
        let json = ch.serialize(&payload);
        assert!(json.contains("Disk near capacity"));
    }

    #[test]
    fn test_opsgenie_serialize_priority() {
        let ch = OpsGenieChannel::new(OpsGenieConfig::new("key"));
        let payload = NotificationPayload::new("Alert", NotifySeverity::Critical);
        let json = ch.serialize(&payload);
        assert!(json.contains("\"P1\""));
    }

    #[test]
    fn test_opsgenie_serialize_warning_priority() {
        let ch = OpsGenieChannel::new(OpsGenieConfig::new("key"));
        let payload = NotificationPayload::new("Alert", NotifySeverity::Warning);
        let json = ch.serialize(&payload);
        assert!(json.contains("\"P3\""));
    }

    #[test]
    fn test_opsgenie_serialize_with_default_tags() {
        let cfg = OpsGenieConfig::new("key")
            .with_tag("service:media")
            .with_tag("env:prod");
        let ch = OpsGenieChannel::new(cfg);
        let payload = NotificationPayload::new("Alert", NotifySeverity::Warning);
        let json = ch.serialize(&payload);
        assert!(json.contains("service:media"));
        assert!(json.contains("env:prod"));
    }

    #[test]
    fn test_opsgenie_serialize_label_as_tag() {
        let ch = OpsGenieChannel::new(OpsGenieConfig::new("key"));
        let payload = NotificationPayload::new("Alert", NotifySeverity::Info)
            .with_label("region", "us-east-1");
        let json = ch.serialize(&payload);
        assert!(json.contains("region:us-east-1"));
    }

    #[test]
    fn test_opsgenie_serialize_dedup_key_as_alias() {
        let ch = OpsGenieChannel::new(OpsGenieConfig::new("key"));
        let payload = NotificationPayload::new("Alert", NotifySeverity::Critical)
            .with_dedup_key("my-dedup-key-789");
        let json = ch.serialize(&payload);
        assert!(json.contains("my-dedup-key-789"));
    }

    #[test]
    fn test_opsgenie_serialize_with_responders() {
        let cfg = OpsGenieConfig::new("key")
            .with_responder("media-ops-team")
            .with_responder("sre-team");
        let ch = OpsGenieChannel::new(cfg);
        let payload = NotificationPayload::new("Alert", NotifySeverity::Critical);
        let json = ch.serialize(&payload);
        assert!(json.contains("media-ops-team"));
        assert!(json.contains("sre-team"));
        assert!(json.contains("\"team\""));
    }

    #[test]
    fn test_opsgenie_serialize_details_include_source_component() {
        let ch = OpsGenieChannel::new(OpsGenieConfig::new("key"));
        let payload = NotificationPayload::new("Alert", NotifySeverity::Warning)
            .with_source("encoder-1")
            .with_component("bitrate-controller");
        let json = ch.serialize(&payload);
        assert!(json.contains("encoder-1"));
        assert!(json.contains("bitrate-controller"));
    }

    #[test]
    fn test_opsgenie_config_default_eu_endpoint() {
        let cfg = OpsGenieConfig::new("key");
        assert!(cfg.endpoint.contains("opsgenie.com"));
        assert!(cfg.endpoint.contains("eu"));
    }

    #[test]
    fn test_opsgenie_config_custom_endpoint() {
        let cfg = OpsGenieConfig::new("key").with_endpoint("http://mock:9090/opsgenie");
        assert_eq!(cfg.endpoint, "http://mock:9090/opsgenie");
    }

    // -- PagerDutyAction --

    #[test]
    fn test_pagerduty_action_str() {
        assert_eq!(PagerDutyAction::Trigger.as_str(), "trigger");
        assert_eq!(PagerDutyAction::Acknowledge.as_str(), "acknowledge");
        assert_eq!(PagerDutyAction::Resolve.as_str(), "resolve");
    }

    // -- json helpers --

    #[test]
    fn test_json_str_escaping() {
        assert_eq!(json_str("hello"), "\"hello\"");
        assert_eq!(json_str("say \"hi\""), "\"say \\\"hi\\\"\"");
    }

    #[test]
    fn test_json_field() {
        let f = json_field("key", "value");
        assert!(f.contains("\"key\""));
        assert!(f.contains("\"value\""));
    }

    #[test]
    fn test_json_object_empty() {
        let obj = json_object(&[]);
        assert_eq!(obj, "{}");
    }

    #[test]
    fn test_json_object_single_field() {
        let obj = json_object(&[json_field("a", "1")]);
        assert!(obj.starts_with('{'));
        assert!(obj.ends_with('}'));
        assert!(obj.contains("\"a\""));
    }

    // -- Integration scenario: critical alert roundtrip --

    #[test]
    fn test_pagerduty_critical_alert_roundtrip() {
        let cfg = PagerDutyConfig::new("integration-key-xyz").with_timeout_ms(5000);
        let ch = PagerDutyChannel::new(cfg);
        let payload = NotificationPayload::new(
            "CRITICAL: encoder throughput dropped below threshold",
            NotifySeverity::Critical,
        )
        .with_source("media-encoder-cluster")
        .with_component("av1-encoder")
        .with_details("Throughput: 18fps (threshold: 23.976fps)")
        .with_label("pipeline", "live-4k")
        .with_label("datacenter", "eu-central-1")
        .with_dedup_key("encoder-throughput-alert-2024");

        let json = ch.serialize(&payload, PagerDutyAction::Trigger);

        // Verify all key fields are present.
        assert!(json.contains("integration-key-xyz"));
        assert!(json.contains("trigger"));
        assert!(json.contains("critical"));
        assert!(json.contains("encoder throughput dropped"));
        assert!(json.contains("media-encoder-cluster"));
        assert!(json.contains("av1-encoder"));
        assert!(json.contains("eu-central-1"));
        assert!(json.contains("encoder-throughput-alert-2024"));
    }

    #[test]
    fn test_opsgenie_critical_alert_roundtrip() {
        let cfg = OpsGenieConfig::new("opsgenie-api-key-abc")
            .with_responder("media-sre")
            .with_tag("service:live-encoding")
            .with_tag("env:production");
        let ch = OpsGenieChannel::new(cfg);
        let payload = NotificationPayload::new(
            "CRITICAL: Live stream buffer underrun detected",
            NotifySeverity::Critical,
        )
        .with_source("ingest-server-3")
        .with_details("Buffer underrun on channel HD-NEWS-42")
        .with_label("channel", "HD-NEWS-42")
        .with_dedup_key("buffer-underrun-ingest-3-hd42");

        let json = ch.serialize(&payload);

        assert!(json.contains("P1"));
        assert!(json.contains("Live stream buffer underrun"));
        assert!(json.contains("media-sre"));
        assert!(json.contains("service:live-encoding"));
        assert!(json.contains("HD-NEWS-42"));
        assert!(json.contains("buffer-underrun-ingest-3-hd42"));
    }
}
