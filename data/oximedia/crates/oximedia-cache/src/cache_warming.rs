//! Predictive cache warming via access-pattern analysis.
//!
//! Records per-key access history, computes frequency/recency metrics, detects
//! periodic access patterns with auto-correlation, and produces a prioritised
//! [`WarmupPlan`] that fits within a given memory budget.

/// Historical access record for a single cache key.
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// The cache key this record belongs to.
    pub key: String,
    /// Monotonically increasing Unix timestamps (seconds) of past accesses.
    pub access_times: Vec<u64>,
    /// Caller-supplied estimated byte size for this key's value.
    pub size_bytes: usize,
}

impl AccessPattern {
    /// Compute accesses per hour over the entire recorded history.
    ///
    /// Returns `0.0` when fewer than two access timestamps are known (the
    /// time-span is undefined).
    pub fn frequency_per_hour(&self) -> f64 {
        if self.access_times.len() < 2 {
            return 0.0;
        }
        let first = *self.access_times.first().unwrap_or(&0);
        let last = *self.access_times.last().unwrap_or(&0);
        let span_secs = last.saturating_sub(first);
        if span_secs == 0 {
            return 0.0;
        }
        let span_hours = span_secs as f64 / 3600.0;
        self.access_times.len() as f64 / span_hours
    }

    /// Predict the next access timestamp using exponential inter-arrival
    /// smoothing (EMA with α = 0.3 on inter-arrival deltas).
    ///
    /// Returns `None` when there are fewer than two data points.
    pub fn predict_next_access(&self) -> Option<u64> {
        if self.access_times.len() < 2 {
            return None;
        }
        // Build inter-arrival sequence.
        let intervals: Vec<f64> = self
            .access_times
            .windows(2)
            .map(|w| w[1].saturating_sub(w[0]) as f64)
            .collect();

        // EMA with α = 0.3 (more weight on recent intervals).
        const ALPHA: f64 = 0.3;
        let mut ema = intervals[0];
        for &interval in intervals.iter().skip(1) {
            ema = ALPHA * interval + (1.0 - ALPHA) * ema;
        }

        let last = *self.access_times.last()?;
        // Round to nearest second, guard against overflow.
        let predicted = last.saturating_add(ema.round().max(0.0) as u64);
        Some(predicted)
    }

    /// Attempt to detect a dominant periodic inter-arrival time (in seconds)
    /// using normalised auto-correlation of the inter-arrival sequence.
    ///
    /// Returns `Some(period)` when the highest off-zero auto-correlation peak
    /// exceeds 0.3 (weak threshold to be liberal about detection).  Returns
    /// `None` when the sequence is too short or no clear periodicity is found.
    pub fn periodicity_secs(&self) -> Option<f64> {
        if self.access_times.len() < 4 {
            return None;
        }
        let intervals: Vec<f64> = self
            .access_times
            .windows(2)
            .map(|w| w[1].saturating_sub(w[0]) as f64)
            .collect();

        let n = intervals.len();
        if n < 3 {
            return None;
        }

        // Compute mean.
        let mean = intervals.iter().sum::<f64>() / n as f64;
        // Compute variance (denominator for normalisation).
        let variance: f64 = intervals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        if variance < 1e-10 {
            // All intervals identical → trivially periodic at that value.
            return Some(mean);
        }

        // Auto-correlation for lags 1..n/2
        let max_lag = (n / 2).max(1);
        let mut best_lag = 0usize;
        let mut best_corr: f64 = 0.0;

        for lag in 1..=max_lag {
            let pairs = n - lag;
            if pairs == 0 {
                break;
            }
            let corr: f64 = (0..pairs)
                .map(|i| (intervals[i] - mean) * (intervals[i + lag] - mean))
                .sum::<f64>()
                / (pairs as f64 * variance);

            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        if best_corr > 0.3 && best_lag > 0 {
            // Estimate the period as the mean inter-arrival time at that lag.
            let period_sum: f64 = (0..(n - best_lag)).map(|i| intervals[i + best_lag]).sum();
            let period = period_sum / (n - best_lag) as f64;
            Some(period)
        } else {
            None
        }
    }
}

// ── CacheWarmer ───────────────────────────────────────────────────────────────

/// A `WarmupPlan` produced by [`CacheWarmer::plan_warmup`].
#[derive(Debug, Clone)]
pub struct WarmupPlan {
    /// Keys to pre-load, ordered by descending priority.
    pub entries_to_warm: Vec<String>,
    /// Total byte cost of all entries in the plan.
    pub estimated_bytes: usize,
    /// Estimated improvement in hit rate (0.0–1.0).
    pub estimated_hit_improvement: f64,
}

/// Predictive cache warmer.
pub struct CacheWarmer {
    /// All recorded access patterns, keyed by cache key.
    pub patterns: Vec<AccessPattern>,
    /// Look-ahead window: only warm entries whose predicted next access is
    /// within this many seconds of `current_time`.
    pub look_ahead_secs: u64,
    /// Minimum accesses/hour for a key to be considered worth warming.
    pub min_frequency: f64,
}

impl CacheWarmer {
    /// Create a new `CacheWarmer` with sensible defaults.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            look_ahead_secs: 300, // 5 minutes
            min_frequency: 1.0,
        }
    }

    /// Record an access to `key` at `time` (Unix seconds).
    ///
    /// If no pattern for this key exists yet, one is created.
    pub fn record_access(&mut self, key: &str, size_bytes: usize, time: u64) {
        if let Some(p) = self.patterns.iter_mut().find(|p| p.key == key) {
            p.size_bytes = size_bytes;
            p.access_times.push(time);
        } else {
            self.patterns.push(AccessPattern {
                key: key.to_string(),
                access_times: vec![time],
                size_bytes,
            });
        }
    }

    /// Build a [`WarmupPlan`] prioritising entries by:
    ///
    /// ```text
    /// score = frequency_per_hour × recency_weight × size_efficiency
    /// ```
    ///
    /// where
    /// * `recency_weight = exp(-age_hours / 1.0)` – exponential decay over 1 h
    /// * `size_efficiency = 1.0 / (1.0 + size_bytes / 1024)`
    ///
    /// Only entries whose predicted next access is within `look_ahead_secs`
    /// of `current_time` and whose frequency exceeds `min_frequency` are
    /// considered.  Entries are added in descending score order until
    /// `available_bytes` would be exceeded.
    pub fn plan_warmup(&self, current_time: u64, available_bytes: usize) -> WarmupPlan {
        // Score each qualifying pattern.
        let mut scored: Vec<(&AccessPattern, f64)> = self
            .patterns
            .iter()
            .filter_map(|p| {
                let freq = p.frequency_per_hour();
                if freq < self.min_frequency {
                    return None;
                }
                // Check predicted next access window.
                let next = p.predict_next_access()?;
                let deadline = current_time.saturating_add(self.look_ahead_secs);
                if next > deadline {
                    return None;
                }
                // Recency weight: based on time since last access.
                let last_access = p.access_times.last().copied().unwrap_or(0);
                let age_secs = current_time.saturating_sub(last_access);
                let age_hours = age_secs as f64 / 3600.0;
                let recency = (-age_hours).exp(); // e^(-age_hours)
                                                  // Size efficiency: smaller entries are cheaper to warm.
                let size_eff = 1.0 / (1.0 + p.size_bytes as f64 / 1024.0);
                let score = freq * recency * size_eff;
                Some((p, score))
            })
            .collect();

        // Sort descending by score.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut entries_to_warm = Vec::new();
        let mut estimated_bytes = 0usize;

        for (pattern, _score) in &scored {
            if estimated_bytes + pattern.size_bytes > available_bytes {
                break;
            }
            estimated_bytes += pattern.size_bytes;
            entries_to_warm.push(pattern.key.clone());
        }

        // Estimate hit improvement as fraction of qualifying entries included.
        let total_qualifying = scored.len();
        let included = entries_to_warm.len();
        let estimated_hit_improvement = if total_qualifying == 0 {
            0.0
        } else {
            included as f64 / total_qualifying as f64
        };

        WarmupPlan {
            entries_to_warm,
            estimated_bytes,
            estimated_hit_improvement,
        }
    }

    /// Return the top `n` hot keys sorted by descending frequency.
    pub fn top_hot_keys(&self, n: usize) -> Vec<(&str, f64)> {
        let mut scored: Vec<(&str, f64)> = self
            .patterns
            .iter()
            .map(|p| (p.key.as_str(), p.frequency_per_hour()))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. record_access creates a new pattern
    #[test]
    fn test_record_access_creates_pattern() {
        let mut warmer = CacheWarmer::new();
        warmer.record_access("key1", 512, 1_000_000);
        assert_eq!(warmer.patterns.len(), 1);
        assert_eq!(warmer.patterns[0].key, "key1");
    }

    // 2. record_access accumulates times for the same key
    #[test]
    fn test_record_access_accumulates() {
        let mut warmer = CacheWarmer::new();
        warmer.record_access("k", 100, 1000);
        warmer.record_access("k", 100, 2000);
        warmer.record_access("k", 100, 3000);
        assert_eq!(warmer.patterns[0].access_times.len(), 3);
    }

    // 3. frequency_per_hour basic
    #[test]
    fn test_frequency_per_hour() {
        let p = AccessPattern {
            key: "k".into(),
            // 6 accesses over 1 hour (3600 s) → ~6 /h
            access_times: vec![0, 720, 1440, 2160, 2880, 3600],
            size_bytes: 128,
        };
        let freq = p.frequency_per_hour();
        assert!((freq - 6.0).abs() < 0.01, "expected ~6/h, got {freq}");
    }

    // 4. frequency_per_hour with fewer than 2 data points
    #[test]
    fn test_frequency_single_point() {
        let p = AccessPattern {
            key: "k".into(),
            access_times: vec![1000],
            size_bytes: 64,
        };
        assert_eq!(p.frequency_per_hour(), 0.0);
    }

    // 5. predict_next_access with uniform intervals
    #[test]
    fn test_predict_next_access_uniform() {
        let p = AccessPattern {
            key: "k".into(),
            // 100 s intervals
            access_times: vec![1000, 1100, 1200, 1300],
            size_bytes: 64,
        };
        let predicted = p.predict_next_access().expect("should predict");
        // With EMA the predicted interval should be close to 100 s.
        assert!(
            predicted >= 1380 && predicted <= 1420,
            "expected ~1400, got {predicted}"
        );
    }

    // 6. predict_next_access returns None with < 2 points
    #[test]
    fn test_predict_next_access_insufficient() {
        let p = AccessPattern {
            key: "k".into(),
            access_times: vec![500],
            size_bytes: 32,
        };
        assert!(p.predict_next_access().is_none());
    }

    // 7. periodicity_secs detects clear period
    #[test]
    fn test_periodicity_detected() {
        // Access every 600 s (10 min) — very regular
        let times: Vec<u64> = (0..20).map(|i| i * 600).collect();
        let p = AccessPattern {
            key: "k".into(),
            access_times: times,
            size_bytes: 64,
        };
        let period = p.periodicity_secs();
        assert!(period.is_some(), "should detect periodicity");
        let period = period.expect("period present");
        assert!(
            (period - 600.0).abs() < 5.0,
            "expected ~600 s, got {period}"
        );
    }

    // 8. periodicity_secs returns None for < 4 points
    #[test]
    fn test_periodicity_too_few_points() {
        let p = AccessPattern {
            key: "k".into(),
            access_times: vec![0, 100, 200],
            size_bytes: 64,
        };
        // Should not panic; may return None or a value — just test no panic and
        // that with exactly 3 points we get None.
        assert!(p.periodicity_secs().is_none());
    }

    // 9. top_hot_keys returns correct order
    #[test]
    fn test_top_hot_keys_order() {
        let mut warmer = CacheWarmer::new();
        // "cold": 2 accesses over 1 hour → ~2/h
        for t in [0u64, 3600] {
            warmer.record_access("cold", 64, t);
        }
        // "hot": 10 accesses over 1 hour → ~10/h
        for i in 0..10u64 {
            warmer.record_access("hot", 64, i * 360);
        }
        let top = warmer.top_hot_keys(2);
        assert_eq!(top[0].0, "hot");
        assert_eq!(top[1].0, "cold");
    }

    // 10. top_hot_keys respects n limit
    #[test]
    fn test_top_hot_keys_limit() {
        let mut warmer = CacheWarmer::new();
        for k in ["a", "b", "c", "d", "e"] {
            warmer.record_access(k, 64, 0);
            warmer.record_access(k, 64, 3600);
        }
        assert_eq!(warmer.top_hot_keys(3).len(), 3);
    }

    // 11. plan_warmup respects available_bytes
    #[test]
    fn test_plan_warmup_respects_budget() {
        let mut warmer = CacheWarmer::new();
        warmer.look_ahead_secs = 10_000;
        warmer.min_frequency = 0.1;
        let now = 10_000u64;
        // Two keys with regular access patterns.
        for i in 0..5u64 {
            warmer.record_access("big", 5000, i * 1800);
            warmer.record_access("small", 100, i * 1800);
        }
        // Only 200 bytes available → "big" (5000 B) must not be included.
        let plan = warmer.plan_warmup(now, 200);
        assert!(plan.estimated_bytes <= 200);
        assert!(!plan.entries_to_warm.contains(&"big".to_string()));
    }

    // 12. plan_warmup excludes keys below min_frequency
    #[test]
    fn test_plan_warmup_min_frequency_filter() {
        let mut warmer = CacheWarmer::new();
        warmer.look_ahead_secs = 100_000;
        warmer.min_frequency = 100.0; // very high threshold
                                      // Only 2 accesses → frequency < 100/h
        warmer.record_access("rare", 64, 0);
        warmer.record_access("rare", 64, 3600);
        let plan = warmer.plan_warmup(7200, usize::MAX);
        assert!(plan.entries_to_warm.is_empty());
    }

    // 13. estimated_hit_improvement is between 0 and 1
    #[test]
    fn test_estimated_hit_improvement_range() {
        let mut warmer = CacheWarmer::new();
        warmer.look_ahead_secs = 100_000;
        warmer.min_frequency = 0.1;
        for i in 0..5u64 {
            warmer.record_access("k", 100, i * 600);
        }
        let plan = warmer.plan_warmup(3000, usize::MAX);
        assert!(plan.estimated_hit_improvement >= 0.0);
        assert!(plan.estimated_hit_improvement <= 1.0);
    }
}
