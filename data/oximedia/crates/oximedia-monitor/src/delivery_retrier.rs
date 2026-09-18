//! Exponential backoff delivery retrier for webhook/email alert failures.
//!
//! When alert delivery via webhook, email, or other channels fails, this
//! module provides a retry mechanism with exponential backoff and jitter.
//! Failed deliveries are queued and retried with increasing delays to
//! avoid overwhelming downstream services.
//!
//! # Design
//!
//! Each delivery attempt is tracked with a `RetryEntry` that records the
//! number of attempts, next retry time, and the delivery payload.  The
//! `DeliveryRetrier` manages a queue of pending retries and exposes a
//! `poll_ready` method that returns entries whose backoff has elapsed.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the delivery retrier.
#[derive(Debug, Clone)]
pub struct RetrierConfig {
    /// Initial backoff delay after the first failure.
    pub initial_backoff: Duration,
    /// Maximum backoff delay (cap).
    pub max_backoff: Duration,
    /// Backoff multiplier (e.g. 2.0 for exponential doubling).
    pub backoff_multiplier: f64,
    /// Maximum number of retry attempts before giving up.
    pub max_attempts: u32,
    /// Maximum number of entries in the retry queue.
    pub max_queue_size: usize,
    /// Jitter factor (0.0 = no jitter, 1.0 = up to 100% of computed delay).
    pub jitter_factor: f64,
}

impl Default for RetrierConfig {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(300), // 5 minutes
            backoff_multiplier: 2.0,
            max_attempts: 5,
            max_queue_size: 1000,
            jitter_factor: 0.25,
        }
    }
}

impl RetrierConfig {
    /// Set the initial backoff delay.
    #[must_use]
    pub fn with_initial_backoff(mut self, d: Duration) -> Self {
        self.initial_backoff = d;
        self
    }

    /// Set the maximum backoff delay.
    #[must_use]
    pub fn with_max_backoff(mut self, d: Duration) -> Self {
        self.max_backoff = d;
        self
    }

    /// Set the backoff multiplier.
    #[must_use]
    pub fn with_multiplier(mut self, m: f64) -> Self {
        self.backoff_multiplier = m.max(1.0);
        self
    }

    /// Set the maximum retry attempts.
    #[must_use]
    pub fn with_max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = n.max(1);
        self
    }

    /// Set the jitter factor.
    #[must_use]
    pub fn with_jitter(mut self, j: f64) -> Self {
        self.jitter_factor = j.clamp(0.0, 1.0);
        self
    }
}

// ---------------------------------------------------------------------------
// Delivery channel
// ---------------------------------------------------------------------------

/// The type of delivery channel that failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeliveryChannel {
    /// Webhook (HTTP POST).
    Webhook,
    /// Email (SMTP).
    Email,
    /// Slack webhook.
    Slack,
    /// Discord webhook.
    Discord,
    /// SMS.
    Sms,
    /// Generic / custom channel.
    Custom,
}

impl DeliveryChannel {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Webhook => "webhook",
            Self::Email => "email",
            Self::Slack => "slack",
            Self::Discord => "discord",
            Self::Sms => "sms",
            Self::Custom => "custom",
        }
    }
}

// ---------------------------------------------------------------------------
// Retry entry
// ---------------------------------------------------------------------------

/// A single delivery that is pending retry.
#[derive(Debug, Clone)]
pub struct RetryEntry {
    /// Unique delivery identifier.
    pub id: String,
    /// Channel that failed.
    pub channel: DeliveryChannel,
    /// The payload to deliver (e.g. alert JSON).
    pub payload: String,
    /// Destination URL or address.
    pub destination: String,
    /// Number of attempts made so far (starts at 1 for the first failure).
    pub attempt: u32,
    /// When the entry was first enqueued.
    pub first_failure: Instant,
    /// When the next retry should be attempted.
    pub next_retry: Instant,
    /// Last error message.
    pub last_error: String,
}

impl RetryEntry {
    /// Returns `true` if the next retry time has passed.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        Instant::now() >= self.next_retry
    }

    /// Returns `true` if the entry has exceeded the maximum attempt count.
    #[must_use]
    pub fn is_exhausted(&self, max_attempts: u32) -> bool {
        self.attempt >= max_attempts
    }

    /// Age since first failure.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.first_failure.elapsed()
    }
}

/// Outcome after a retry attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryOutcome {
    /// Delivery succeeded; entry removed from queue.
    Delivered,
    /// Delivery failed; entry re-queued with increased backoff.
    Requeued {
        /// Next retry delay.
        next_delay: Duration,
    },
    /// Maximum attempts exhausted; entry dropped.
    Exhausted,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistics for the delivery retrier.
#[derive(Debug, Clone, Copy, Default)]
pub struct RetrierStats {
    /// Total entries enqueued.
    pub total_enqueued: u64,
    /// Total successful deliveries.
    pub total_delivered: u64,
    /// Total entries exhausted (gave up).
    pub total_exhausted: u64,
    /// Total retry attempts.
    pub total_retries: u64,
    /// Total entries dropped due to queue overflow.
    pub total_dropped: u64,
}

// ---------------------------------------------------------------------------
// DeliveryRetrier
// ---------------------------------------------------------------------------

/// Manages failed delivery retries with exponential backoff.
#[derive(Debug)]
pub struct DeliveryRetrier {
    config: RetrierConfig,
    queue: VecDeque<RetryEntry>,
    stats: RetrierStats,
    /// Simple counter for deterministic jitter (avoids dependency on rand).
    jitter_counter: u64,
}

impl DeliveryRetrier {
    /// Create a new retrier with the given configuration.
    #[must_use]
    pub fn new(config: RetrierConfig) -> Self {
        Self {
            config,
            queue: VecDeque::new(),
            stats: RetrierStats::default(),
            jitter_counter: 0,
        }
    }

    /// Create a retrier with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(RetrierConfig::default())
    }

    /// Compute the backoff delay for a given attempt number.
    ///
    /// Uses the formula: `min(initial * multiplier^(attempt-1), max_backoff)` + jitter.
    fn compute_backoff(&mut self, attempt: u32) -> Duration {
        let base_secs = self.config.initial_backoff.as_secs_f64()
            * self
                .config
                .backoff_multiplier
                .powi(attempt.saturating_sub(1) as i32);

        let capped_secs = base_secs.min(self.config.max_backoff.as_secs_f64());

        // Deterministic jitter using a simple LCG.
        let jitter = if self.config.jitter_factor > 0.0 {
            self.jitter_counter = self
                .jitter_counter
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1442695040888963407);
            let frac = (self.jitter_counter >> 33) as f64 / (u32::MAX as f64);
            capped_secs * self.config.jitter_factor * frac
        } else {
            0.0
        };

        Duration::from_secs_f64(capped_secs + jitter)
    }

    /// Enqueue a failed delivery for retry.
    ///
    /// The entry will be scheduled for its first retry after `initial_backoff`.
    pub fn enqueue(
        &mut self,
        id: impl Into<String>,
        channel: DeliveryChannel,
        payload: impl Into<String>,
        destination: impl Into<String>,
        error: impl Into<String>,
    ) {
        // Enforce queue limit.
        if self.queue.len() >= self.config.max_queue_size {
            self.queue.pop_front();
            self.stats.total_dropped += 1;
        }

        let now = Instant::now();
        let delay = self.compute_backoff(1);

        self.queue.push_back(RetryEntry {
            id: id.into(),
            channel,
            payload: payload.into(),
            destination: destination.into(),
            attempt: 1,
            first_failure: now,
            next_retry: now + delay,
            last_error: error.into(),
        });

        self.stats.total_enqueued += 1;
    }

    /// Return all entries whose backoff has elapsed (ready for retry).
    ///
    /// Removes them from the queue. The caller should attempt delivery
    /// and then call `report_success` or `report_failure` for each.
    pub fn poll_ready(&mut self) -> Vec<RetryEntry> {
        let now = Instant::now();
        let mut ready = Vec::new();
        let mut remaining = VecDeque::new();

        for entry in self.queue.drain(..) {
            if now >= entry.next_retry {
                ready.push(entry);
            } else {
                remaining.push_back(entry);
            }
        }

        self.queue = remaining;
        ready
    }

    /// Report that a delivery succeeded. Updates stats.
    pub fn report_success(&mut self, _entry: &RetryEntry) {
        self.stats.total_delivered += 1;
    }

    /// Report that a delivery attempt failed again.
    ///
    /// If the entry has not exhausted its retries, it is re-queued with an
    /// increased backoff. Returns the outcome.
    pub fn report_failure(
        &mut self,
        mut entry: RetryEntry,
        error: impl Into<String>,
    ) -> RetryOutcome {
        self.stats.total_retries += 1;
        entry.attempt += 1;
        entry.last_error = error.into();

        if entry.is_exhausted(self.config.max_attempts) {
            self.stats.total_exhausted += 1;
            return RetryOutcome::Exhausted;
        }

        let delay = self.compute_backoff(entry.attempt);
        entry.next_retry = Instant::now() + delay;

        // Enforce queue limit.
        if self.queue.len() >= self.config.max_queue_size {
            self.queue.pop_front();
            self.stats.total_dropped += 1;
        }

        self.queue.push_back(entry);

        RetryOutcome::Requeued { next_delay: delay }
    }

    /// Number of entries in the retry queue.
    #[must_use]
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` if the retry queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get a snapshot of the retrier statistics.
    #[must_use]
    pub fn stats(&self) -> RetrierStats {
        self.stats
    }

    /// Peek at the next retry time (earliest in queue).
    #[must_use]
    pub fn next_retry_time(&self) -> Option<Instant> {
        self.queue.iter().map(|e| e.next_retry).min()
    }

    /// Remove all entries for a specific delivery ID.
    pub fn cancel(&mut self, id: &str) {
        self.queue.retain(|e| e.id != id);
    }

    /// Clear all pending retries.
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// Reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &RetrierConfig {
        &self.config
    }
}

impl Default for DeliveryRetrier {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- RetrierConfig --

    #[test]
    fn test_config_default() {
        let cfg = RetrierConfig::default();
        assert_eq!(cfg.initial_backoff, Duration::from_secs(1));
        assert_eq!(cfg.max_backoff, Duration::from_secs(300));
        assert!((cfg.backoff_multiplier - 2.0).abs() < 1e-9);
        assert_eq!(cfg.max_attempts, 5);
    }

    #[test]
    fn test_config_builders() {
        let cfg = RetrierConfig::default()
            .with_initial_backoff(Duration::from_millis(500))
            .with_max_backoff(Duration::from_secs(60))
            .with_multiplier(3.0)
            .with_max_attempts(10)
            .with_jitter(0.5);
        assert_eq!(cfg.initial_backoff, Duration::from_millis(500));
        assert_eq!(cfg.max_backoff, Duration::from_secs(60));
        assert!((cfg.backoff_multiplier - 3.0).abs() < 1e-9);
        assert_eq!(cfg.max_attempts, 10);
        assert!((cfg.jitter_factor - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_config_min_multiplier() {
        let cfg = RetrierConfig::default().with_multiplier(0.5);
        assert!((cfg.backoff_multiplier - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_clamp_jitter() {
        let cfg = RetrierConfig::default().with_jitter(5.0);
        assert!((cfg.jitter_factor - 1.0).abs() < 1e-9);
    }

    // -- DeliveryChannel --

    #[test]
    fn test_channel_labels() {
        assert_eq!(DeliveryChannel::Webhook.label(), "webhook");
        assert_eq!(DeliveryChannel::Email.label(), "email");
        assert_eq!(DeliveryChannel::Slack.label(), "slack");
        assert_eq!(DeliveryChannel::Discord.label(), "discord");
        assert_eq!(DeliveryChannel::Sms.label(), "sms");
        assert_eq!(DeliveryChannel::Custom.label(), "custom");
    }

    // -- RetryEntry --

    #[test]
    fn test_entry_is_ready_future() {
        let entry = RetryEntry {
            id: "test".to_string(),
            channel: DeliveryChannel::Webhook,
            payload: "{}".to_string(),
            destination: "http://example.com".to_string(),
            attempt: 1,
            first_failure: Instant::now(),
            next_retry: Instant::now() + Duration::from_secs(3600),
            last_error: "timeout".to_string(),
        };
        assert!(!entry.is_ready());
    }

    #[test]
    fn test_entry_is_ready_past() {
        let entry = RetryEntry {
            id: "test".to_string(),
            channel: DeliveryChannel::Email,
            payload: "alert".to_string(),
            destination: "admin@example.com".to_string(),
            attempt: 1,
            first_failure: Instant::now(),
            next_retry: Instant::now(), // now = ready
            last_error: "connection refused".to_string(),
        };
        assert!(entry.is_ready());
    }

    #[test]
    fn test_entry_is_exhausted() {
        let entry = RetryEntry {
            id: "test".to_string(),
            channel: DeliveryChannel::Webhook,
            payload: String::new(),
            destination: String::new(),
            attempt: 5,
            first_failure: Instant::now(),
            next_retry: Instant::now(),
            last_error: String::new(),
        };
        assert!(entry.is_exhausted(5));
        assert!(!entry.is_exhausted(6));
    }

    // -- DeliveryRetrier basic --

    #[test]
    fn test_retrier_new() {
        let r = DeliveryRetrier::with_defaults();
        assert!(r.is_empty());
        assert_eq!(r.queue_len(), 0);
    }

    #[test]
    fn test_enqueue() {
        let mut r = DeliveryRetrier::with_defaults();
        r.enqueue(
            "a1",
            DeliveryChannel::Webhook,
            "{}",
            "http://x.com",
            "timeout",
        );
        assert_eq!(r.queue_len(), 1);
        assert_eq!(r.stats().total_enqueued, 1);
    }

    #[test]
    fn test_poll_ready_returns_entries_past_deadline() {
        let mut r = DeliveryRetrier::new(
            RetrierConfig::default()
                .with_initial_backoff(Duration::from_millis(0))
                .with_jitter(0.0),
        );
        r.enqueue("a1", DeliveryChannel::Email, "alert", "admin@x.com", "fail");
        let ready = r.poll_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "a1");
        assert!(r.is_empty());
    }

    #[test]
    fn test_poll_ready_skips_future_entries() {
        let mut r = DeliveryRetrier::new(
            RetrierConfig::default()
                .with_initial_backoff(Duration::from_secs(3600))
                .with_jitter(0.0),
        );
        r.enqueue("a1", DeliveryChannel::Webhook, "{}", "http://x.com", "err");
        let ready = r.poll_ready();
        assert!(ready.is_empty());
        assert_eq!(r.queue_len(), 1);
    }

    // -- Success / failure reporting --

    #[test]
    fn test_report_success() {
        let mut r = DeliveryRetrier::new(
            RetrierConfig::default()
                .with_initial_backoff(Duration::from_millis(0))
                .with_jitter(0.0),
        );
        r.enqueue("a1", DeliveryChannel::Slack, "{}", "http://slack", "err");
        let ready = r.poll_ready();
        r.report_success(&ready[0]);
        assert_eq!(r.stats().total_delivered, 1);
        assert!(r.is_empty());
    }

    #[test]
    fn test_report_failure_requeues() {
        let mut r = DeliveryRetrier::new(
            RetrierConfig::default()
                .with_initial_backoff(Duration::from_millis(0))
                .with_max_attempts(3)
                .with_jitter(0.0),
        );
        r.enqueue("a1", DeliveryChannel::Webhook, "{}", "http://x.com", "err");
        let ready = r.poll_ready();
        let outcome = r.report_failure(
            ready.into_iter().next().expect("should have entry"),
            "timeout",
        );
        assert!(matches!(outcome, RetryOutcome::Requeued { .. }));
        assert_eq!(r.queue_len(), 1);
        assert_eq!(r.stats().total_retries, 1);
    }

    #[test]
    fn test_report_failure_exhausted() {
        let mut r = DeliveryRetrier::new(
            RetrierConfig::default()
                .with_initial_backoff(Duration::from_millis(0))
                .with_max_attempts(1)
                .with_jitter(0.0),
        );
        r.enqueue("a1", DeliveryChannel::Webhook, "{}", "http://x.com", "err");
        let ready = r.poll_ready();
        let outcome = r.report_failure(
            ready.into_iter().next().expect("should have entry"),
            "still failing",
        );
        assert_eq!(outcome, RetryOutcome::Exhausted);
        assert!(r.is_empty());
        assert_eq!(r.stats().total_exhausted, 1);
    }

    // -- Exponential backoff --

    #[test]
    fn test_backoff_increases_exponentially() {
        let mut r = DeliveryRetrier::new(
            RetrierConfig::default()
                .with_initial_backoff(Duration::from_secs(1))
                .with_max_backoff(Duration::from_secs(300))
                .with_multiplier(2.0)
                .with_jitter(0.0),
        );

        let d1 = r.compute_backoff(1);
        let d2 = r.compute_backoff(2);
        let d3 = r.compute_backoff(3);

        assert!((d1.as_secs_f64() - 1.0).abs() < 0.01);
        assert!((d2.as_secs_f64() - 2.0).abs() < 0.01);
        assert!((d3.as_secs_f64() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_backoff_capped_at_max() {
        let mut r = DeliveryRetrier::new(
            RetrierConfig::default()
                .with_initial_backoff(Duration::from_secs(1))
                .with_max_backoff(Duration::from_secs(10))
                .with_multiplier(2.0)
                .with_jitter(0.0),
        );

        let d10 = r.compute_backoff(10);
        assert!(d10.as_secs_f64() <= 10.5); // allow small float imprecision
    }

    #[test]
    fn test_backoff_with_jitter_varies() {
        let mut r = DeliveryRetrier::new(
            RetrierConfig::default()
                .with_initial_backoff(Duration::from_secs(1))
                .with_jitter(0.5),
        );

        let d1 = r.compute_backoff(1);
        let d2 = r.compute_backoff(1);
        // With jitter, two calls may produce different results.
        // At minimum they should both be >= initial_backoff.
        assert!(d1.as_secs_f64() >= 1.0);
        assert!(d2.as_secs_f64() >= 1.0);
    }

    // -- Queue management --

    #[test]
    fn test_queue_overflow_drops_oldest() {
        let mut r = DeliveryRetrier::new(
            RetrierConfig::default()
                .with_initial_backoff(Duration::from_secs(3600))
                .with_jitter(0.0),
        );
        r.config.max_queue_size = 2;

        r.enqueue("a1", DeliveryChannel::Webhook, "{}", "http://1", "err");
        r.enqueue("a2", DeliveryChannel::Webhook, "{}", "http://2", "err");
        r.enqueue("a3", DeliveryChannel::Webhook, "{}", "http://3", "err");

        assert_eq!(r.queue_len(), 2);
        assert_eq!(r.stats().total_dropped, 1);
    }

    #[test]
    fn test_cancel() {
        let mut r = DeliveryRetrier::new(
            RetrierConfig::default()
                .with_initial_backoff(Duration::from_secs(3600))
                .with_jitter(0.0),
        );
        r.enqueue("a1", DeliveryChannel::Webhook, "{}", "http://x.com", "err");
        r.enqueue("a2", DeliveryChannel::Webhook, "{}", "http://x.com", "err");
        r.cancel("a1");
        assert_eq!(r.queue_len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut r = DeliveryRetrier::with_defaults();
        r.enqueue("a1", DeliveryChannel::Email, "", "", "");
        r.enqueue("a2", DeliveryChannel::Email, "", "", "");
        r.clear();
        assert!(r.is_empty());
    }

    // -- Full retry lifecycle --

    #[test]
    fn test_full_retry_lifecycle() {
        // max_attempts=4: initial attempt(1) + 3 retries before exhaustion.
        let mut r = DeliveryRetrier::new(
            RetrierConfig::default()
                .with_initial_backoff(Duration::from_millis(0))
                .with_max_attempts(4)
                .with_jitter(0.0),
        );

        r.enqueue(
            "alert_1",
            DeliveryChannel::Webhook,
            "{\"alert\":true}",
            "http://hook.example.com",
            "connection refused",
        );

        // Attempt 1 (initial enqueue counts as attempt 1).
        let ready = r.poll_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].attempt, 1);

        // Fail -> attempt bumped to 2, requeued.
        let outcome = r.report_failure(ready.into_iter().next().expect("entry exists"), "timeout");
        assert!(matches!(outcome, RetryOutcome::Requeued { .. }));

        // Attempt 2.
        let ready = r.poll_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].attempt, 2);

        // Fail -> attempt bumped to 3, requeued.
        let outcome = r.report_failure(ready.into_iter().next().expect("entry exists"), "503");
        assert!(matches!(outcome, RetryOutcome::Requeued { .. }));

        // Attempt 3.
        let ready = r.poll_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].attempt, 3);

        // Fail -> attempt bumped to 4, which equals max_attempts -> exhausted.
        let outcome =
            r.report_failure(ready.into_iter().next().expect("entry exists"), "still 503");
        assert_eq!(outcome, RetryOutcome::Exhausted);
        assert!(r.is_empty());

        let stats = r.stats();
        assert_eq!(stats.total_enqueued, 1);
        assert_eq!(stats.total_exhausted, 1);
        assert_eq!(stats.total_retries, 3);
    }

    #[test]
    fn test_retry_then_success() {
        let mut r = DeliveryRetrier::new(
            RetrierConfig::default()
                .with_initial_backoff(Duration::from_millis(0))
                .with_max_attempts(5)
                .with_jitter(0.0),
        );

        r.enqueue(
            "alert_2",
            DeliveryChannel::Slack,
            "{}",
            "http://slack",
            "err",
        );

        // Fail first attempt.
        let ready = r.poll_ready();
        let outcome = r.report_failure(ready.into_iter().next().expect("entry"), "timeout");
        assert!(matches!(outcome, RetryOutcome::Requeued { .. }));

        // Succeed on second attempt.
        let ready = r.poll_ready();
        r.report_success(&ready[0]);

        assert!(r.is_empty());
        assert_eq!(r.stats().total_delivered, 1);
        assert_eq!(r.stats().total_retries, 1);
    }

    // -- next_retry_time --

    #[test]
    fn test_next_retry_time_empty() {
        let r = DeliveryRetrier::with_defaults();
        assert!(r.next_retry_time().is_none());
    }

    #[test]
    fn test_next_retry_time_populated() {
        let mut r = DeliveryRetrier::new(
            RetrierConfig::default()
                .with_initial_backoff(Duration::from_secs(10))
                .with_jitter(0.0),
        );
        r.enqueue("a1", DeliveryChannel::Webhook, "{}", "http://x", "err");
        let t = r.next_retry_time().expect("should have a time");
        assert!(t > Instant::now());
    }
}
