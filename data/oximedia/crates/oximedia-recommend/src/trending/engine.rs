//! Trending content detection engine.
//!
//! Tracks view events and scores items based on recent popularity using
//! exponential time decay.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

/// A single content view event.
pub struct ViewEvent {
    /// Identifier of the viewed item.
    pub item_id: u64,
    /// Identifier of the viewer.
    pub user_id: u64,
    /// Unix timestamp when the view occurred (seconds).
    pub timestamp: u64,
    /// How long the user watched, in seconds.
    pub duration_s: u32,
}

impl ViewEvent {
    /// Create a new view event.
    #[must_use]
    pub fn new(item_id: u64, user_id: u64, timestamp: u64, duration_s: u32) -> Self {
        Self {
            item_id,
            user_id,
            timestamp,
            duration_s,
        }
    }
}

/// Aggregated trending score for a single item.
pub struct TrendScore {
    /// Item identifier.
    pub item_id: u64,
    /// Composite trending score.
    pub score: f64,
    /// Total (decay-weighted) view count.
    pub view_count: u64,
    /// Number of unique viewers.
    pub unique_viewers: u64,
    /// Average watch duration across all views.
    pub avg_duration_s: f64,
}

impl TrendScore {
    /// Return the engagement rate: `avg_duration_s / views`.
    ///
    /// Returns `0.0` when `views` is zero.
    #[must_use]
    pub fn engagement_rate(&self, views: u64) -> f64 {
        if views == 0 {
            return 0.0;
        }
        self.avg_duration_s / views as f64
    }
}

/// Exponential time-decay weight.
///
/// Returns a value in `(0, 1]`.  When `event_age_s == 0` the weight is `1.0`.
/// The weight halves every `half_life_s` seconds.  If `half_life_s` is zero
/// the weight is always `1.0`.
#[must_use]
pub fn time_decay_weight(event_age_s: u64, half_life_s: u64) -> f64 {
    if half_life_s == 0 {
        return 1.0;
    }
    let t = event_age_s as f64 / half_life_s as f64;
    (-t * std::f64::consts::LN_2).exp()
}

/// An engine that tracks view events and computes trending scores.
pub struct TrendingEngine {
    /// All recorded view events.
    pub events: Vec<ViewEvent>,
    /// Half-life for time-decay weighting (seconds).
    pub decay_half_life_s: u64,
}

impl TrendingEngine {
    /// Create a new engine with the specified decay half-life.
    #[must_use]
    pub fn new(decay_half_life_s: u64) -> Self {
        Self {
            events: Vec::new(),
            decay_half_life_s,
        }
    }

    /// Record a new view event.
    pub fn record_view(&mut self, event: ViewEvent) {
        self.events.push(event);
    }

    /// Compute trending scores for all items, using `now` as the current time.
    #[must_use]
    pub fn compute_scores(&self, now: u64) -> Vec<TrendScore> {
        // Aggregate per-item stats.
        let mut weighted_views: HashMap<u64, f64> = HashMap::new();
        let mut unique_viewers: HashMap<u64, HashSet<u64>> = HashMap::new();
        let mut total_duration: HashMap<u64, f64> = HashMap::new();
        let mut raw_view_count: HashMap<u64, u64> = HashMap::new();

        for event in &self.events {
            let age = now.saturating_sub(event.timestamp);
            let w = time_decay_weight(age, self.decay_half_life_s);

            *weighted_views.entry(event.item_id).or_insert(0.0) += w;
            unique_viewers
                .entry(event.item_id)
                .or_default()
                .insert(event.user_id);
            *total_duration.entry(event.item_id).or_insert(0.0) += f64::from(event.duration_s);
            *raw_view_count.entry(event.item_id).or_insert(0) += 1;
        }

        weighted_views
            .into_iter()
            .map(|(item_id, wv)| {
                let uv = unique_viewers.get(&item_id).map_or(0, |s| s.len() as u64);
                let rc = *raw_view_count.get(&item_id).unwrap_or(&0);
                let avg_dur = if rc == 0 {
                    0.0
                } else {
                    *total_duration.get(&item_id).unwrap_or(&0.0) / rc as f64
                };
                // Score: decay-weighted views * log(unique_viewers + 1)
                let score = wv * (uv as f64 + 1.0).ln().max(1.0);
                TrendScore {
                    item_id,
                    score,
                    view_count: wv as u64,
                    unique_viewers: uv,
                    avg_duration_s: avg_dur,
                }
            })
            .collect()
    }

    /// Return the top `n` trending items, sorted by descending score.
    #[must_use]
    pub fn top_trending(&self, n: usize, now: u64) -> Vec<TrendScore> {
        let mut scores = self.compute_scores(now);
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(n);
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_event_new() {
        let e = ViewEvent::new(1, 2, 1000, 30);
        assert_eq!(e.item_id, 1);
        assert_eq!(e.user_id, 2);
        assert_eq!(e.timestamp, 1000);
        assert_eq!(e.duration_s, 30);
    }

    #[test]
    fn test_time_decay_weight_zero_age() {
        assert!((time_decay_weight(0, 3600) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_time_decay_weight_half_life() {
        let w = time_decay_weight(3600, 3600);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_time_decay_weight_zero_half_life() {
        assert!((time_decay_weight(9999, 0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_time_decay_weight_two_half_lives() {
        let w = time_decay_weight(7200, 3600);
        assert!((w - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_trending_engine_new() {
        let e = TrendingEngine::new(3600);
        assert_eq!(e.decay_half_life_s, 3600);
        assert!(e.events.is_empty());
    }

    #[test]
    fn test_record_view_and_compute() {
        let mut engine = TrendingEngine::new(3600);
        let now = 10_000u64;
        engine.record_view(ViewEvent::new(1, 1, now, 60));
        engine.record_view(ViewEvent::new(1, 2, now, 90));
        let scores = engine.compute_scores(now);
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].item_id, 1);
        assert_eq!(scores[0].unique_viewers, 2);
    }

    #[test]
    fn test_top_trending_ordering() {
        let mut engine = TrendingEngine::new(3600);
        let now = 10_000u64;
        // item 2 gets more recent views
        for u in 0..5u64 {
            engine.record_view(ViewEvent::new(2, u, now, 30));
        }
        engine.record_view(ViewEvent::new(1, 0, now, 30));
        let top = engine.top_trending(2, now);
        assert_eq!(top[0].item_id, 2);
    }

    #[test]
    fn test_top_trending_empty() {
        let engine = TrendingEngine::new(3600);
        assert!(engine.top_trending(5, 0).is_empty());
    }

    #[test]
    fn test_trend_score_engagement_rate() {
        let ts = TrendScore {
            item_id: 1,
            score: 1.0,
            view_count: 10,
            unique_viewers: 5,
            avg_duration_s: 45.0,
        };
        assert!((ts.engagement_rate(5) - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_trend_score_engagement_zero_views() {
        let ts = TrendScore {
            item_id: 1,
            score: 0.0,
            view_count: 0,
            unique_viewers: 0,
            avg_duration_s: 0.0,
        };
        assert!((ts.engagement_rate(0)).abs() < 1e-9);
    }

    #[test]
    fn test_decay_reduces_old_events() {
        let mut engine = TrendingEngine::new(1); // 1-second half-life
        let now = 100u64;
        // Event 100 seconds ago → nearly zero weight
        engine.record_view(ViewEvent::new(1, 1, 0, 60));
        // Recent event
        engine.record_view(ViewEvent::new(2, 2, 100, 60));
        let top = engine.top_trending(2, now);
        // item 2 should have a higher score
        assert_eq!(top[0].item_id, 2);
    }
}
