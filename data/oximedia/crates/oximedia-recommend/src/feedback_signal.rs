//! Feedback signal processing for recommendation systems.
//!
//! This module handles different types of user feedback signals
//! (clicks, watches, likes, shares, etc.) and aggregates them
//! into unified preference scores for recommendation training.

#![allow(dead_code)]

use std::collections::HashMap;
use uuid::Uuid;

/// Types of feedback signals that can be collected from users.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SignalType {
    /// User clicked on content
    Click,
    /// User watched content (partial or full)
    Watch,
    /// User explicitly liked content
    Like,
    /// User explicitly disliked content
    Dislike,
    /// User shared content externally
    Share,
    /// User saved/bookmarked content
    Bookmark,
    /// User skipped content in a feed
    Skip,
    /// User added content to a playlist
    AddToPlaylist,
    /// User rated content (1-5 stars)
    ExplicitRating,
    /// User hovered/previewed content
    Preview,
}

impl SignalType {
    /// Returns the default weight for this signal type.
    ///
    /// Higher weights indicate stronger preference signals.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn default_weight(&self) -> f64 {
        match self {
            Self::Click => 0.3,
            Self::Watch => 0.6,
            Self::Like => 0.8,
            Self::Dislike => -0.9,
            Self::Share => 0.9,
            Self::Bookmark => 0.7,
            Self::Skip => -0.2,
            Self::AddToPlaylist => 0.75,
            Self::ExplicitRating => 1.0,
            Self::Preview => 0.1,
        }
    }

    /// Returns true if this signal indicates positive preference.
    #[must_use]
    pub fn is_positive(&self) -> bool {
        self.default_weight() > 0.0
    }
}

/// A single feedback signal event from a user.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeedbackSignal {
    /// Unique ID for this signal event
    pub id: Uuid,
    /// User who generated the signal
    pub user_id: Uuid,
    /// Content the signal is about
    pub content_id: Uuid,
    /// Type of signal
    pub signal_type: SignalType,
    /// Signal value (e.g., watch percentage 0.0-1.0, rating 1.0-5.0)
    pub value: f64,
    /// Unix timestamp when the signal was generated
    pub timestamp: i64,
    /// Optional context metadata
    pub context: HashMap<String, String>,
}

impl FeedbackSignal {
    /// Creates a new feedback signal.
    #[must_use]
    pub fn new(user_id: Uuid, content_id: Uuid, signal_type: SignalType, value: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            user_id,
            content_id,
            signal_type,
            value,
            timestamp: chrono::Utc::now().timestamp(),
            context: HashMap::new(),
        }
    }

    /// Creates a new feedback signal with context.
    #[must_use]
    pub fn with_context(mut self, key: &str, val: &str) -> Self {
        self.context.insert(key.to_string(), val.to_string());
        self
    }

    /// Computes the weighted score for this signal.
    ///
    /// The weighted score combines the signal type's default weight
    /// with the raw signal value.
    #[must_use]
    pub fn weighted_score(&self) -> f64 {
        self.signal_type.default_weight() * self.value
    }

    /// Returns the age of the signal in seconds relative to a reference timestamp.
    #[must_use]
    pub fn age_seconds(&self, now: i64) -> i64 {
        now - self.timestamp
    }
}

/// Aggregated signal statistics for a user-content pair.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AggregatedSignal {
    /// User ID
    pub user_id: Uuid,
    /// Content ID
    pub content_id: Uuid,
    /// Aggregated preference score
    pub score: f64,
    /// Number of signals contributing to the score
    pub signal_count: usize,
    /// Timestamp of the most recent signal
    pub last_signal_at: i64,
}

/// Aggregates multiple feedback signals into unified preference scores.
///
/// The aggregator collects signals over time and computes a single
/// preference score per user-content pair using configurable weights
/// and time-decay functions.
#[derive(Debug)]
pub struct SignalAggregator {
    /// Custom weights per signal type (overrides defaults)
    custom_weights: HashMap<SignalType, f64>,
    /// Half-life for time decay in seconds
    decay_half_life_secs: i64,
    /// Stored signals indexed by (`user_id`, `content_id`)
    signals: HashMap<(Uuid, Uuid), Vec<FeedbackSignal>>,
}

impl SignalAggregator {
    /// Creates a new signal aggregator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            custom_weights: HashMap::new(),
            decay_half_life_secs: 7 * 24 * 3600, // 7 days
            signals: HashMap::new(),
        }
    }

    /// Creates a new aggregator with a custom decay half-life.
    #[must_use]
    pub fn with_decay_half_life(half_life_secs: i64) -> Self {
        Self {
            custom_weights: HashMap::new(),
            decay_half_life_secs: half_life_secs,
            signals: HashMap::new(),
        }
    }

    /// Sets a custom weight for a specific signal type.
    pub fn set_weight(&mut self, signal_type: SignalType, weight: f64) {
        self.custom_weights.insert(signal_type, weight);
    }

    /// Gets the effective weight for a signal type.
    #[must_use]
    pub fn effective_weight(&self, signal_type: &SignalType) -> f64 {
        self.custom_weights
            .get(signal_type)
            .copied()
            .unwrap_or_else(|| signal_type.default_weight())
    }

    /// Ingests a feedback signal into the aggregator.
    pub fn ingest(&mut self, signal: FeedbackSignal) {
        let key = (signal.user_id, signal.content_id);
        self.signals.entry(key).or_default().push(signal);
    }

    /// Computes the time-decay factor for a signal.
    ///
    /// Uses exponential decay: `exp(-ln(2) * age / half_life)`.
    #[must_use]
    pub fn time_decay(&self, age_secs: i64) -> f64 {
        if age_secs <= 0 {
            return 1.0;
        }
        let lambda = (2.0_f64).ln() / self.decay_half_life_secs as f64;
        (-lambda * age_secs as f64).exp()
    }

    /// Aggregates signals for a specific user-content pair.
    ///
    /// Returns `None` if no signals exist for the pair.
    #[must_use]
    pub fn aggregate(&self, user_id: Uuid, content_id: Uuid, now: i64) -> Option<AggregatedSignal> {
        let key = (user_id, content_id);
        let signals = self.signals.get(&key)?;
        if signals.is_empty() {
            return None;
        }

        let mut total_score = 0.0;
        let mut last_ts = i64::MIN;

        for sig in signals {
            let weight = self.effective_weight(&sig.signal_type);
            let decay = self.time_decay(sig.age_seconds(now));
            total_score += weight * sig.value * decay;
            if sig.timestamp > last_ts {
                last_ts = sig.timestamp;
            }
        }

        Some(AggregatedSignal {
            user_id,
            content_id,
            score: total_score,
            signal_count: signals.len(),
            last_signal_at: last_ts,
        })
    }

    /// Aggregates all signals for a given user, returning scores per content.
    #[must_use]
    pub fn aggregate_for_user(&self, user_id: Uuid, now: i64) -> Vec<AggregatedSignal> {
        self.signals
            .keys()
            .filter(|(uid, _)| *uid == user_id)
            .filter_map(|(_, cid)| self.aggregate(user_id, *cid, now))
            .collect()
    }

    /// Returns the total number of stored signal events.
    #[must_use]
    pub fn total_signals(&self) -> usize {
        self.signals.values().map(Vec::len).sum()
    }

    /// Removes all signals older than the given cutoff timestamp.
    pub fn evict_before(&mut self, cutoff_ts: i64) {
        for signals in self.signals.values_mut() {
            signals.retain(|s| s.timestamp >= cutoff_ts);
        }
        self.signals.retain(|_, v| !v.is_empty());
    }
}

impl Default for SignalAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user() -> Uuid {
        Uuid::new_v4()
    }
    fn content() -> Uuid {
        Uuid::new_v4()
    }

    #[test]
    fn test_signal_type_default_weights() {
        assert!(SignalType::Click.default_weight() > 0.0);
        assert!(SignalType::Dislike.default_weight() < 0.0);
        assert!(SignalType::Skip.default_weight() < 0.0);
        assert!((SignalType::ExplicitRating.default_weight() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_signal_type_is_positive() {
        assert!(SignalType::Like.is_positive());
        assert!(SignalType::Share.is_positive());
        assert!(!SignalType::Dislike.is_positive());
        assert!(!SignalType::Skip.is_positive());
    }

    #[test]
    fn test_feedback_signal_creation() {
        let u = user();
        let c = content();
        let sig = FeedbackSignal::new(u, c, SignalType::Watch, 0.75);
        assert_eq!(sig.user_id, u);
        assert_eq!(sig.content_id, c);
        assert!((sig.value - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_feedback_signal_with_context() {
        let sig = FeedbackSignal::new(user(), content(), SignalType::Click, 1.0)
            .with_context("device", "mobile")
            .with_context("source", "feed");
        assert_eq!(sig.context.len(), 2);
        assert_eq!(
            sig.context.get("device").expect("should succeed in test"),
            "mobile"
        );
    }

    #[test]
    fn test_weighted_score() {
        let sig = FeedbackSignal::new(user(), content(), SignalType::Like, 1.0);
        let ws = sig.weighted_score();
        assert!((ws - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weighted_score_partial_watch() {
        let sig = FeedbackSignal::new(user(), content(), SignalType::Watch, 0.5);
        let ws = sig.weighted_score();
        assert!((ws - 0.3).abs() < f64::EPSILON); // 0.6 * 0.5
    }

    #[test]
    fn test_age_seconds() {
        let mut sig = FeedbackSignal::new(user(), content(), SignalType::Click, 1.0);
        sig.timestamp = 1000;
        assert_eq!(sig.age_seconds(1500), 500);
    }

    #[test]
    fn test_aggregator_ingest_and_count() {
        let mut agg = SignalAggregator::new();
        let u = user();
        let c = content();
        agg.ingest(FeedbackSignal::new(u, c, SignalType::Click, 1.0));
        agg.ingest(FeedbackSignal::new(u, c, SignalType::Watch, 0.8));
        assert_eq!(agg.total_signals(), 2);
    }

    #[test]
    fn test_aggregator_aggregate_basic() {
        let mut agg = SignalAggregator::new();
        let u = user();
        let c = content();
        let now = chrono::Utc::now().timestamp();
        let mut sig = FeedbackSignal::new(u, c, SignalType::Like, 1.0);
        sig.timestamp = now;
        agg.ingest(sig);
        let result = agg.aggregate(u, c, now).expect("should succeed in test");
        assert_eq!(result.signal_count, 1);
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_aggregator_no_signals() {
        let agg = SignalAggregator::new();
        let result = agg.aggregate(user(), content(), 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_time_decay_zero_age() {
        let agg = SignalAggregator::new();
        assert!((agg.time_decay(0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_decay_half_life() {
        let agg = SignalAggregator::with_decay_half_life(100);
        let decay = agg.time_decay(100);
        assert!((decay - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_custom_weight_override() {
        let mut agg = SignalAggregator::new();
        agg.set_weight(SignalType::Click, 0.99);
        assert!((agg.effective_weight(&SignalType::Click) - 0.99).abs() < f64::EPSILON);
        // Non-overridden still uses default
        assert!(
            (agg.effective_weight(&SignalType::Like) - SignalType::Like.default_weight()).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_evict_before() {
        let mut agg = SignalAggregator::new();
        let u = user();
        let c = content();
        let mut old = FeedbackSignal::new(u, c, SignalType::Click, 1.0);
        old.timestamp = 100;
        let mut recent = FeedbackSignal::new(u, c, SignalType::Watch, 0.9);
        recent.timestamp = 500;
        agg.ingest(old);
        agg.ingest(recent);
        assert_eq!(agg.total_signals(), 2);
        agg.evict_before(300);
        assert_eq!(agg.total_signals(), 1);
    }

    #[test]
    fn test_aggregate_for_user() {
        let mut agg = SignalAggregator::new();
        let u = user();
        let c1 = content();
        let c2 = content();
        let now = chrono::Utc::now().timestamp();
        let mut s1 = FeedbackSignal::new(u, c1, SignalType::Like, 1.0);
        s1.timestamp = now;
        let mut s2 = FeedbackSignal::new(u, c2, SignalType::Share, 1.0);
        s2.timestamp = now;
        agg.ingest(s1);
        agg.ingest(s2);
        let results = agg.aggregate_for_user(u, now);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_negative_signal_reduces_score() {
        let mut agg = SignalAggregator::new();
        let u = user();
        let c = content();
        let now = chrono::Utc::now().timestamp();
        let mut s1 = FeedbackSignal::new(u, c, SignalType::Like, 1.0);
        s1.timestamp = now;
        let mut s2 = FeedbackSignal::new(u, c, SignalType::Dislike, 1.0);
        s2.timestamp = now;
        agg.ingest(s1);
        agg.ingest(s2);
        let result = agg.aggregate(u, c, now).expect("should succeed in test");
        // Like (0.8) + Dislike (-0.9) = -0.1
        assert!(result.score < 0.0);
    }
}
