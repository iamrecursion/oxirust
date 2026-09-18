//! Backpressure and rate-limiting for distributed encoding pipelines.
//!
//! Provides token-bucket rate limiting, credit-based flow control, and a
//! backpressure signal aggregator to prevent upstream producers from
//! overwhelming downstream consumers.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// TokenBucket
// ---------------------------------------------------------------------------

/// A token-bucket rate limiter.
///
/// Tokens accumulate at `refill_rate` tokens per millisecond up to `capacity`.
/// Each request consumes `tokens` tokens.  Requests that cannot be served
/// immediately are rejected (non-blocking).
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Maximum number of tokens the bucket can hold.
    pub capacity: f64,
    /// Token refill rate in tokens per millisecond.
    pub refill_rate: f64,
    /// Current number of available tokens.
    tokens: f64,
    /// Unix epoch ms of the last refill computation.
    last_refill_ms: u64,
}

impl TokenBucket {
    /// Create a new token bucket, starting full.
    #[must_use]
    pub fn new(capacity: f64, refill_rate: f64, now_ms: u64) -> Self {
        Self {
            capacity,
            refill_rate,
            tokens: capacity,
            last_refill_ms: now_ms,
        }
    }

    /// Refill tokens based on elapsed time since last refill.
    fn refill(&mut self, now_ms: u64) {
        if now_ms <= self.last_refill_ms {
            return;
        }
        let elapsed = (now_ms - self.last_refill_ms) as f64;
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill_ms = now_ms;
    }

    /// Try to consume `tokens` from the bucket.
    ///
    /// Returns `true` if the tokens were available and consumed, `false`
    /// otherwise (no tokens are consumed on failure).
    pub fn try_consume(&mut self, tokens: f64, now_ms: u64) -> bool {
        self.refill(now_ms);
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    /// Current available tokens (after refilling to `now_ms`).
    #[must_use]
    pub fn available(&mut self, now_ms: u64) -> f64 {
        self.refill(now_ms);
        self.tokens
    }

    /// Returns `true` if the bucket is completely full.
    #[must_use]
    pub fn is_full(&mut self, now_ms: u64) -> bool {
        self.refill(now_ms);
        (self.tokens - self.capacity).abs() < f64::EPSILON
    }
}

// ---------------------------------------------------------------------------
// CreditAccount
// ---------------------------------------------------------------------------

/// Credit-based flow control for a single producer-consumer pair.
///
/// The consumer grants credits to the producer; the producer sends one unit
/// of work per credit and decrements the credit balance.
#[derive(Debug, Clone)]
pub struct CreditAccount {
    /// Current credit balance (number of items the producer may send).
    credits: i64,
    /// Maximum credits the consumer will grant at once.
    pub max_credits: i64,
}

impl CreditAccount {
    /// Create a new credit account with zero balance.
    #[must_use]
    pub fn new(max_credits: i64) -> Self {
        Self {
            credits: 0,
            max_credits,
        }
    }

    /// Consumer grants `n` credits (clamped so balance never exceeds `max_credits`).
    pub fn grant(&mut self, n: i64) {
        self.credits = (self.credits + n).min(self.max_credits);
    }

    /// Producer consumes one credit to send one unit of work.
    ///
    /// Returns `true` if a credit was available, `false` if the producer
    /// should pause.
    pub fn consume(&mut self) -> bool {
        if self.credits > 0 {
            self.credits -= 1;
            true
        } else {
            false
        }
    }

    /// Current credit balance.
    #[must_use]
    pub fn balance(&self) -> i64 {
        self.credits
    }

    /// Returns `true` if the producer may send work.
    #[must_use]
    pub fn may_send(&self) -> bool {
        self.credits > 0
    }
}

// ---------------------------------------------------------------------------
// BackpressureSignal
// ---------------------------------------------------------------------------

/// The current backpressure state of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BackpressureLevel {
    /// No backpressure; producer may run at full rate.
    None,
    /// Mild backpressure; producer should throttle slightly.
    Low,
    /// Moderate backpressure; producer should throttle significantly.
    Medium,
    /// Severe backpressure; producer should pause.
    High,
    /// Critical backpressure; producer must stop immediately.
    Critical,
}

impl BackpressureLevel {
    /// Recommended rate multiplier (fraction of nominal rate to apply).
    ///
    /// `1.0` = no throttling; `0.0` = stop completely.
    #[must_use]
    pub fn rate_multiplier(&self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Low => 0.75,
            Self::Medium => 0.5,
            Self::High => 0.2,
            Self::Critical => 0.0,
        }
    }

    /// Derive a backpressure level from a queue fill ratio (0.0–1.0).
    #[must_use]
    pub fn from_fill_ratio(ratio: f64) -> Self {
        let ratio = ratio.clamp(0.0, 1.0);
        if ratio < 0.5 {
            Self::None
        } else if ratio < 0.7 {
            Self::Low
        } else if ratio < 0.85 {
            Self::Medium
        } else if ratio < 0.95 {
            Self::High
        } else {
            Self::Critical
        }
    }
}

// ---------------------------------------------------------------------------
// NodeBackpressure
// ---------------------------------------------------------------------------

/// Backpressure signal from a single node.
#[derive(Debug, Clone)]
pub struct NodeBackpressure {
    /// Node identifier.
    pub node_id: String,
    /// Current backpressure level.
    pub level: BackpressureLevel,
    /// Queue fill ratio that led to this level (0.0–1.0).
    pub fill_ratio: f64,
    /// Timestamp when this signal was recorded (Unix epoch ms).
    pub timestamp_ms: u64,
}

impl NodeBackpressure {
    /// Create a node backpressure signal from a fill ratio.
    #[must_use]
    pub fn from_fill(node_id: impl Into<String>, fill_ratio: f64, timestamp_ms: u64) -> Self {
        Self {
            node_id: node_id.into(),
            level: BackpressureLevel::from_fill_ratio(fill_ratio),
            fill_ratio,
            timestamp_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// BackpressureAggregator
// ---------------------------------------------------------------------------

/// Aggregates backpressure signals from multiple nodes and computes a
/// cluster-wide backpressure recommendation.
#[derive(Debug, Default)]
pub struct BackpressureAggregator {
    signals: Vec<NodeBackpressure>,
}

impl BackpressureAggregator {
    /// Create an empty aggregator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record or update a node's backpressure signal.
    pub fn update(&mut self, signal: NodeBackpressure) {
        if let Some(existing) = self
            .signals
            .iter_mut()
            .find(|s| s.node_id == signal.node_id)
        {
            *existing = signal;
        } else {
            self.signals.push(signal);
        }
    }

    /// Remove signals older than `now_ms - ttl_ms`.
    pub fn evict_stale(&mut self, now_ms: u64, ttl_ms: u64) {
        let cutoff = now_ms.saturating_sub(ttl_ms);
        self.signals.retain(|s| s.timestamp_ms >= cutoff);
    }

    /// The maximum (worst) backpressure level across all nodes.
    #[must_use]
    pub fn max_level(&self) -> BackpressureLevel {
        self.signals
            .iter()
            .map(|s| s.level)
            .max()
            .unwrap_or(BackpressureLevel::None)
    }

    /// Mean queue fill ratio across all nodes.
    #[must_use]
    pub fn mean_fill_ratio(&self) -> f64 {
        if self.signals.is_empty() {
            return 0.0;
        }
        self.signals.iter().map(|s| s.fill_ratio).sum::<f64>() / self.signals.len() as f64
    }

    /// Recommended cluster-wide rate multiplier (minimum across all nodes).
    #[must_use]
    pub fn recommended_rate_multiplier(&self) -> f64 {
        self.signals
            .iter()
            .map(|s| s.level.rate_multiplier())
            .fold(1.0_f64, f64::min)
    }

    /// Number of nodes reporting at or above `level`.
    #[must_use]
    pub fn count_at_or_above(&self, level: BackpressureLevel) -> usize {
        self.signals.iter().filter(|s| s.level >= level).count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── TokenBucket ──────────────────────────────────────────────────────

    #[test]
    fn test_token_bucket_starts_full() {
        let mut bucket = TokenBucket::new(100.0, 1.0, 0);
        assert!((bucket.available(0) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_token_bucket_consume_success() {
        let mut bucket = TokenBucket::new(100.0, 1.0, 0);
        assert!(bucket.try_consume(50.0, 0));
        assert!((bucket.available(0) - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_token_bucket_consume_fail_insufficient() {
        let mut bucket = TokenBucket::new(10.0, 0.0, 0);
        assert!(!bucket.try_consume(20.0, 0));
        // Tokens unchanged
        assert!((bucket.available(0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_token_bucket_refills_over_time() {
        let mut bucket = TokenBucket::new(100.0, 10.0, 0); // 10 tokens/ms
        bucket.try_consume(100.0, 0); // drain
        assert!(bucket.available(0) < 1.0);
        let avail = bucket.available(5); // 5 ms later → +50 tokens
        assert!((avail - 50.0).abs() < 1e-6, "avail={avail}");
    }

    #[test]
    fn test_token_bucket_does_not_exceed_capacity() {
        let mut bucket = TokenBucket::new(50.0, 100.0, 0);
        let avail = bucket.available(1000);
        assert!((avail - 50.0).abs() < 1e-9, "avail={avail}");
    }

    #[test]
    fn test_token_bucket_is_full_initially() {
        let mut bucket = TokenBucket::new(10.0, 1.0, 0);
        assert!(bucket.is_full(0));
    }

    // ── CreditAccount ────────────────────────────────────────────────────

    #[test]
    fn test_credit_account_starts_at_zero() {
        let account = CreditAccount::new(100);
        assert_eq!(account.balance(), 0);
    }

    #[test]
    fn test_credit_account_grant_increases_balance() {
        let mut account = CreditAccount::new(100);
        account.grant(10);
        assert_eq!(account.balance(), 10);
    }

    #[test]
    fn test_credit_account_grant_capped_at_max() {
        let mut account = CreditAccount::new(5);
        account.grant(100);
        assert_eq!(account.balance(), 5);
    }

    #[test]
    fn test_credit_account_consume_success() {
        let mut account = CreditAccount::new(10);
        account.grant(3);
        assert!(account.consume());
        assert_eq!(account.balance(), 2);
    }

    #[test]
    fn test_credit_account_consume_fail_when_empty() {
        let mut account = CreditAccount::new(10);
        assert!(!account.consume());
    }

    #[test]
    fn test_credit_account_may_send() {
        let mut account = CreditAccount::new(10);
        assert!(!account.may_send());
        account.grant(1);
        assert!(account.may_send());
    }

    // ── BackpressureLevel ────────────────────────────────────────────────

    #[test]
    fn test_backpressure_from_fill_ratio_none() {
        assert_eq!(
            BackpressureLevel::from_fill_ratio(0.0),
            BackpressureLevel::None
        );
        assert_eq!(
            BackpressureLevel::from_fill_ratio(0.49),
            BackpressureLevel::None
        );
    }

    #[test]
    fn test_backpressure_from_fill_ratio_low() {
        assert_eq!(
            BackpressureLevel::from_fill_ratio(0.6),
            BackpressureLevel::Low
        );
    }

    #[test]
    fn test_backpressure_from_fill_ratio_medium() {
        assert_eq!(
            BackpressureLevel::from_fill_ratio(0.75),
            BackpressureLevel::Medium
        );
    }

    #[test]
    fn test_backpressure_from_fill_ratio_high() {
        assert_eq!(
            BackpressureLevel::from_fill_ratio(0.9),
            BackpressureLevel::High
        );
    }

    #[test]
    fn test_backpressure_from_fill_ratio_critical() {
        assert_eq!(
            BackpressureLevel::from_fill_ratio(1.0),
            BackpressureLevel::Critical
        );
    }

    #[test]
    fn test_backpressure_rate_multiplier_none() {
        assert!((BackpressureLevel::None.rate_multiplier() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_backpressure_rate_multiplier_critical() {
        assert!(BackpressureLevel::Critical.rate_multiplier() < f64::EPSILON);
    }

    // ── BackpressureAggregator ───────────────────────────────────────────

    #[test]
    fn test_aggregator_empty_max_level_is_none() {
        let agg = BackpressureAggregator::new();
        assert_eq!(agg.max_level(), BackpressureLevel::None);
    }

    #[test]
    fn test_aggregator_max_level_selects_worst() {
        let mut agg = BackpressureAggregator::new();
        agg.update(NodeBackpressure::from_fill("n0", 0.3, 1000));
        agg.update(NodeBackpressure::from_fill("n1", 0.9, 1000));
        assert_eq!(agg.max_level(), BackpressureLevel::High);
    }

    #[test]
    fn test_aggregator_mean_fill_ratio() {
        let mut agg = BackpressureAggregator::new();
        agg.update(NodeBackpressure::from_fill("n0", 0.4, 1000));
        agg.update(NodeBackpressure::from_fill("n1", 0.6, 1000));
        let mean = agg.mean_fill_ratio();
        assert!((mean - 0.5).abs() < 1e-9, "mean={mean}");
    }

    #[test]
    fn test_aggregator_recommended_rate_minimum() {
        let mut agg = BackpressureAggregator::new();
        agg.update(NodeBackpressure::from_fill("n0", 0.2, 1000)); // None → 1.0
        agg.update(NodeBackpressure::from_fill("n1", 0.8, 1000)); // Medium → 0.5
                                                                  // Minimum is 0.5
        let rate = agg.recommended_rate_multiplier();
        assert!((rate - 0.5).abs() < 1e-6, "rate={rate}");
    }

    #[test]
    fn test_aggregator_evict_stale() {
        let mut agg = BackpressureAggregator::new();
        agg.update(NodeBackpressure::from_fill("n0", 0.5, 100));
        agg.update(NodeBackpressure::from_fill("n1", 0.5, 5000));
        agg.evict_stale(5000, 1000); // cutoff = 4000 → n0 removed
        assert_eq!(agg.count_at_or_above(BackpressureLevel::None), 1);
    }

    #[test]
    fn test_aggregator_count_at_or_above() {
        let mut agg = BackpressureAggregator::new();
        agg.update(NodeBackpressure::from_fill("n0", 0.2, 1000)); // None
        agg.update(NodeBackpressure::from_fill("n1", 0.6, 1000)); // Low
        agg.update(NodeBackpressure::from_fill("n2", 0.9, 1000)); // High
        assert_eq!(agg.count_at_or_above(BackpressureLevel::Low), 2);
        assert_eq!(agg.count_at_or_above(BackpressureLevel::High), 1);
    }
}
