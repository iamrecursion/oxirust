// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Notification system for workflow events.
//!
//! Provides a rule-based dispatcher that matches events and severities to
//! channels, renders templates with variable substitution, and records every
//! dispatch for auditing.

/// Dispatch channel classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispatchChannel {
    /// Electronic mail.
    Email,
    /// HTTP(S) webhook endpoint.
    Webhook,
    /// Slack workspace message.
    Slack,
    /// SMS short message.
    Sms,
    /// In-application notification.
    InApp,
}

impl DispatchChannel {
    /// Returns `true` for channels that involve external network calls.
    #[must_use]
    pub const fn is_external(self) -> bool {
        !matches!(self, Self::InApp)
    }

    /// Typical round-trip latency expectation in milliseconds.
    #[must_use]
    pub const fn typical_latency_ms(self) -> u32 {
        match self {
            Self::InApp => 5,
            Self::Webhook => 200,
            Self::Slack => 300,
            Self::Email => 5_000,
            Self::Sms => 2_000,
        }
    }
}

/// A message template with `{{key}}` placeholder syntax.
#[derive(Debug, Clone)]
pub struct NotificationTemplate {
    /// Short subject line.
    pub subject: String,
    /// Body text; may contain `{{key}}` placeholders.
    pub body_template: String,
}

impl NotificationTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(subject: impl Into<String>, body_template: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            body_template: body_template.into(),
        }
    }

    /// Render the body by substituting every `{{key}}` with its value from `vars`.
    ///
    /// Unrecognised placeholders are left unchanged.
    #[must_use]
    pub fn render(&self, vars: &[(String, String)]) -> String {
        let mut result = self.body_template.clone();
        for (key, value) in vars {
            let placeholder = format!("{{{{{key}}}}}");
            result = result.replace(&placeholder, value);
        }
        result
    }
}

/// Severity level for a notification event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NotificationSeverity {
    /// Informational message.
    Info,
    /// Non-critical warning.
    Warning,
    /// Recoverable error.
    Error,
    /// Unrecoverable / urgent error.
    Critical,
}

impl NotificationSeverity {
    /// Numeric level: higher is more severe (Info=0, Critical=3).
    #[must_use]
    pub const fn level(self) -> u8 {
        match self {
            Self::Info => 0,
            Self::Warning => 1,
            Self::Error => 2,
            Self::Critical => 3,
        }
    }
}

/// A rule that maps a trigger + minimum severity to one or more channels.
#[derive(Debug, Clone)]
pub struct NotificationRule {
    /// Event name that activates this rule.
    pub trigger: String,
    /// Channels to dispatch on when matched.
    pub channels: Vec<DispatchChannel>,
    /// Template to render for this rule.
    pub template: NotificationTemplate,
    /// Minimum severity level required to dispatch.
    pub min_severity: NotificationSeverity,
}

impl NotificationRule {
    /// Create a new rule.
    #[must_use]
    pub fn new(
        trigger: impl Into<String>,
        channels: Vec<DispatchChannel>,
        template: NotificationTemplate,
        min_severity: NotificationSeverity,
    ) -> Self {
        Self {
            trigger: trigger.into(),
            channels,
            template,
            min_severity,
        }
    }

    /// Returns `true` when `trigger` matches and `severity >= min_severity`.
    #[must_use]
    pub fn matches(&self, trigger: &str, severity: NotificationSeverity) -> bool {
        self.trigger == trigger && severity >= self.min_severity
    }
}

/// Accumulates rules and records dispatched notifications.
#[derive(Debug, Default)]
pub struct NotificationDispatcher {
    /// Registered rules.
    pub rules: Vec<NotificationRule>,
    /// Audit log: `(timestamp_ms, rendered_message)`.
    pub sent: Vec<(u64, String)>,
}

impl NotificationDispatcher {
    /// Create an empty dispatcher.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a rule.
    pub fn add_rule(&mut self, r: NotificationRule) {
        self.rules.push(r);
    }

    /// Evaluate all rules for `trigger` + `severity`.
    ///
    /// For every matching rule every channel is "dispatched" (recorded in the
    /// audit log). Returns the total number of channel dispatches performed.
    pub fn dispatch(
        &mut self,
        trigger: &str,
        severity: NotificationSeverity,
        vars: &[(String, String)],
        now_ms: u64,
    ) -> usize {
        let mut count = 0;

        // Collect matching (rendered_body, channel_count) pairs first to
        // avoid borrow issues.
        let dispatches: Vec<(String, usize)> = self
            .rules
            .iter()
            .filter(|r| r.matches(trigger, severity))
            .map(|r| {
                let body = r.template.render(vars);
                (body, r.channels.len())
            })
            .collect();

        for (body, ch_count) in dispatches {
            for _ in 0..ch_count {
                self.sent.push((now_ms, body.clone()));
                count += 1;
            }
        }

        count
    }

    /// Total number of channel dispatches recorded since creation.
    #[must_use]
    pub fn total_sent(&self) -> usize {
        self.sent.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- DispatchChannel ---

    #[test]
    fn test_in_app_is_not_external() {
        assert!(!DispatchChannel::InApp.is_external());
    }

    #[test]
    fn test_external_channels() {
        assert!(DispatchChannel::Email.is_external());
        assert!(DispatchChannel::Webhook.is_external());
        assert!(DispatchChannel::Slack.is_external());
        assert!(DispatchChannel::Sms.is_external());
    }

    #[test]
    fn test_typical_latency_ordering() {
        assert!(
            DispatchChannel::InApp.typical_latency_ms()
                < DispatchChannel::Webhook.typical_latency_ms()
        );
        assert!(
            DispatchChannel::Sms.typical_latency_ms() < DispatchChannel::Email.typical_latency_ms()
        );
    }

    // --- NotificationTemplate ---

    #[test]
    fn test_render_replaces_placeholder() {
        let t = NotificationTemplate::new("Subject", "Hello {{name}}!");
        let rendered = t.render(&[("name".to_string(), "World".to_string())]);
        assert_eq!(rendered, "Hello World!");
    }

    #[test]
    fn test_render_multiple_placeholders() {
        let t = NotificationTemplate::new("S", "{{a}} and {{b}}");
        let rendered = t.render(&[
            ("a".to_string(), "foo".to_string()),
            ("b".to_string(), "bar".to_string()),
        ]);
        assert_eq!(rendered, "foo and bar");
    }

    #[test]
    fn test_render_unknown_placeholder_unchanged() {
        let t = NotificationTemplate::new("S", "Hello {{unknown}}");
        let rendered = t.render(&[("name".to_string(), "X".to_string())]);
        assert_eq!(rendered, "Hello {{unknown}}");
    }

    #[test]
    fn test_render_no_vars() {
        let t = NotificationTemplate::new("S", "plain text");
        assert_eq!(t.render(&[]), "plain text");
    }

    // --- NotificationSeverity ---

    #[test]
    fn test_severity_levels() {
        assert_eq!(NotificationSeverity::Info.level(), 0);
        assert_eq!(NotificationSeverity::Warning.level(), 1);
        assert_eq!(NotificationSeverity::Error.level(), 2);
        assert_eq!(NotificationSeverity::Critical.level(), 3);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(NotificationSeverity::Info < NotificationSeverity::Warning);
        assert!(NotificationSeverity::Warning < NotificationSeverity::Error);
        assert!(NotificationSeverity::Error < NotificationSeverity::Critical);
    }

    // --- NotificationRule ---

    #[test]
    fn test_rule_matches_exact() {
        let t = NotificationTemplate::new("S", "B");
        let r = NotificationRule::new(
            "job_done",
            vec![DispatchChannel::InApp],
            t,
            NotificationSeverity::Info,
        );
        assert!(r.matches("job_done", NotificationSeverity::Info));
    }

    #[test]
    fn test_rule_matches_higher_severity() {
        let t = NotificationTemplate::new("S", "B");
        let r = NotificationRule::new(
            "job_done",
            vec![DispatchChannel::InApp],
            t,
            NotificationSeverity::Info,
        );
        assert!(r.matches("job_done", NotificationSeverity::Critical));
    }

    #[test]
    fn test_rule_does_not_match_lower_severity() {
        let t = NotificationTemplate::new("S", "B");
        let r = NotificationRule::new(
            "job_done",
            vec![DispatchChannel::Email],
            t,
            NotificationSeverity::Error,
        );
        assert!(!r.matches("job_done", NotificationSeverity::Warning));
    }

    #[test]
    fn test_rule_does_not_match_different_trigger() {
        let t = NotificationTemplate::new("S", "B");
        let r = NotificationRule::new(
            "job_done",
            vec![DispatchChannel::Email],
            t,
            NotificationSeverity::Info,
        );
        assert!(!r.matches("job_failed", NotificationSeverity::Critical));
    }

    // --- NotificationDispatcher ---

    fn make_dispatcher() -> NotificationDispatcher {
        let mut d = NotificationDispatcher::new();
        let t = NotificationTemplate::new("Alert", "Job {{job}} finished.");
        d.add_rule(NotificationRule::new(
            "job_complete",
            vec![DispatchChannel::Email, DispatchChannel::Slack],
            t,
            NotificationSeverity::Info,
        ));
        d
    }

    #[test]
    fn test_dispatch_returns_channel_count() {
        let mut d = make_dispatcher();
        let vars = vec![("job".to_string(), "encode_001".to_string())];
        let sent = d.dispatch("job_complete", NotificationSeverity::Info, &vars, 1000);
        assert_eq!(sent, 2); // Email + Slack
    }

    #[test]
    fn test_dispatch_no_match_returns_zero() {
        let mut d = make_dispatcher();
        let sent = d.dispatch("unknown_event", NotificationSeverity::Critical, &[], 1000);
        assert_eq!(sent, 0);
    }

    #[test]
    fn test_total_sent_accumulates() {
        let mut d = make_dispatcher();
        d.dispatch("job_complete", NotificationSeverity::Info, &[], 1000);
        d.dispatch("job_complete", NotificationSeverity::Warning, &[], 2000);
        assert_eq!(d.total_sent(), 4); // 2 channels × 2 dispatches
    }

    #[test]
    fn test_dispatch_renders_template() {
        let mut d = make_dispatcher();
        let vars = vec![("job".to_string(), "my-job".to_string())];
        d.dispatch("job_complete", NotificationSeverity::Info, &vars, 5000);
        assert!(d.sent[0].1.contains("my-job"));
    }
}
