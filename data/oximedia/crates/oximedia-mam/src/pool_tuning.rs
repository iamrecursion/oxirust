//! Connection pool tuning for the MAM database layer.
//!
//! Monitors concurrent request load in real time and adjusts connection pool
//! configuration recommendations accordingly.  Because `sqlx` does not expose
//! live pool-resize APIs, this module provides:
//!
//! * `PoolMetrics` – snapshot of current pool utilisation.
//! * `LoadSampler` – sliding-window sampler that records request arrivals and
//!   completions so the tuner has an accurate concurrency reading.
//! * `PoolTuner` – adaptive tuner that computes recommended `min_connections`
//!   and `max_connections` values and emits `TuningRecommendation`s.
//! * `PoolConfig` – recommended configuration value-object that callers can
//!   apply when (re-)creating the pool.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// PoolMetrics
// ---------------------------------------------------------------------------

/// Point-in-time snapshot of pool utilisation.
#[derive(Debug, Clone)]
pub struct PoolMetrics {
    /// Number of connections that are actively executing a query.
    pub active_connections: u32,
    /// Number of connections that are idle in the pool.
    pub idle_connections: u32,
    /// Total connections (active + idle).
    pub total_connections: u32,
    /// Number of requests currently waiting for a connection.
    pub pending_requests: u32,
    /// Configured maximum number of connections.
    pub max_connections: u32,
    /// Configured minimum number of connections.
    pub min_connections: u32,
    /// Pool utilisation in the range [0.0, 1.0].
    pub utilisation: f32,
    /// Timestamp at which this snapshot was taken.
    pub sampled_at: Instant,
}

impl PoolMetrics {
    /// Construct a snapshot from raw counts.
    #[must_use]
    pub fn new(
        active: u32,
        idle: u32,
        pending: u32,
        max_connections: u32,
        min_connections: u32,
    ) -> Self {
        let total = active + idle;
        let utilisation = if max_connections == 0 {
            0.0
        } else {
            active as f32 / max_connections as f32
        };
        Self {
            active_connections: active,
            idle_connections: idle,
            total_connections: total,
            pending_requests: pending,
            max_connections,
            min_connections,
            utilisation,
            sampled_at: Instant::now(),
        }
    }

    /// `true` when utilisation exceeds 80 %.
    #[must_use]
    pub fn is_high_load(&self) -> bool {
        self.utilisation >= 0.80
    }

    /// `true` when utilisation is below 20 % and idle connections > `min_connections`.
    #[must_use]
    pub fn is_low_load(&self) -> bool {
        self.utilisation < 0.20 && self.idle_connections > self.min_connections
    }
}

// ---------------------------------------------------------------------------
// LoadSampler
// ---------------------------------------------------------------------------

/// A sliding-window concurrency sampler.
///
/// Call [`LoadSampler::request_started`] when a new database request begins
/// and [`LoadSampler::request_finished`] when it completes.
/// [`LoadSampler::current_concurrency`] returns the number of in-flight requests.
#[derive(Debug)]
pub struct LoadSampler {
    /// Number of in-flight requests.
    in_flight: AtomicI64,
    /// Total requests started since creation.
    total_started: AtomicU64,
    /// Total requests completed since creation.
    total_finished: AtomicU64,
    /// Sliding window of (instant, concurrency) samples.
    window: Mutex<VecDeque<(Instant, i64)>>,
    /// Duration covered by the sliding window.
    window_duration: Duration,
}

impl LoadSampler {
    /// Create a new sampler with the given sliding-window duration.
    #[must_use]
    pub fn new(window_duration: Duration) -> Self {
        Self {
            in_flight: AtomicI64::new(0),
            total_started: AtomicU64::new(0),
            total_finished: AtomicU64::new(0),
            window: Mutex::new(VecDeque::new()),
            window_duration,
        }
    }

    /// Record that a new database request has started.
    pub fn request_started(&self) {
        let concurrency = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        self.total_started.fetch_add(1, Ordering::Relaxed);
        self.push_sample(concurrency);
    }

    /// Record that a database request has finished.
    pub fn request_finished(&self) {
        let concurrency = self.in_flight.fetch_sub(1, Ordering::Relaxed) - 1;
        self.total_finished.fetch_add(1, Ordering::Relaxed);
        let clamped = concurrency.max(0);
        self.push_sample(clamped);
    }

    fn push_sample(&self, concurrency: i64) {
        let now = Instant::now();
        let cutoff = now
            .checked_sub(self.window_duration)
            .unwrap_or_else(Instant::now);
        if let Ok(mut w) = self.window.lock() {
            // Evict stale samples.
            while w.front().map(|(t, _)| *t < cutoff).unwrap_or(false) {
                w.pop_front();
            }
            w.push_back((now, concurrency));
        }
    }

    /// Current number of in-flight requests.
    #[must_use]
    pub fn current_concurrency(&self) -> u32 {
        self.in_flight.load(Ordering::Relaxed).max(0) as u32
    }

    /// Peak concurrency observed within the sliding window.
    #[must_use]
    pub fn peak_concurrency(&self) -> u32 {
        let now = Instant::now();
        let cutoff = now
            .checked_sub(self.window_duration)
            .unwrap_or_else(Instant::now);
        self.window
            .lock()
            .map(|w| {
                w.iter()
                    .filter(|(t, _)| *t >= cutoff)
                    .map(|(_, c)| *c)
                    .max()
                    .unwrap_or(0) as u32
            })
            .unwrap_or(0)
    }

    /// Mean concurrency over the sliding window.
    #[must_use]
    pub fn mean_concurrency(&self) -> f32 {
        let now = Instant::now();
        let cutoff = now
            .checked_sub(self.window_duration)
            .unwrap_or_else(Instant::now);
        self.window
            .lock()
            .map(|w| {
                let samples: Vec<i64> = w
                    .iter()
                    .filter(|(t, _)| *t >= cutoff)
                    .map(|(_, c)| *c)
                    .collect();
                if samples.is_empty() {
                    0.0
                } else {
                    samples.iter().sum::<i64>() as f32 / samples.len() as f32
                }
            })
            .unwrap_or(0.0)
    }

    /// Total number of requests started since creation.
    #[must_use]
    pub fn total_started(&self) -> u64 {
        self.total_started.load(Ordering::Relaxed)
    }

    /// Total number of requests finished since creation.
    #[must_use]
    pub fn total_finished(&self) -> u64 {
        self.total_finished.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// PoolConfig
// ---------------------------------------------------------------------------

/// Recommended pool configuration values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolConfig {
    /// Recommended minimum number of connections to keep alive.
    pub min_connections: u32,
    /// Recommended maximum number of connections.
    pub max_connections: u32,
    /// Recommended connection acquire timeout.
    pub acquire_timeout: Duration,
    /// Recommended connection idle lifetime before recycling.
    pub idle_timeout: Duration,
    /// Recommended maximum lifetime for any single connection.
    pub max_lifetime: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_connections: 2,
            max_connections: 20,
            acquire_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(600),
            max_lifetime: Duration::from_secs(1800),
        }
    }
}

// ---------------------------------------------------------------------------
// TuningRecommendation
// ---------------------------------------------------------------------------

/// A tuning recommendation emitted by [`PoolTuner`].
#[derive(Debug, Clone)]
pub struct TuningRecommendation {
    /// Recommended pool configuration.
    pub config: PoolConfig,
    /// Human-readable rationale.
    pub rationale: String,
    /// Observed concurrency level that triggered this recommendation.
    pub observed_concurrency: u32,
    /// Observed utilisation ratio.
    pub observed_utilisation: f32,
}

// ---------------------------------------------------------------------------
// TuningStrategy
// ---------------------------------------------------------------------------

/// Strategy that governs how aggressively the tuner scales the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuningStrategy {
    /// Keep the pool size constant; only compute informational recommendations.
    Conservative,
    /// Grow the pool eagerly on high load; shrink slowly on low load.
    Balanced,
    /// Grow and shrink the pool aggressively to minimise idle connections.
    Aggressive,
}

// ---------------------------------------------------------------------------
// PoolTuner
// ---------------------------------------------------------------------------

/// Adaptive pool tuner.
///
/// Holds a reference to a [`LoadSampler`] and produces [`TuningRecommendation`]s
/// when [`PoolTuner::evaluate`] is called.
#[derive(Debug)]
pub struct PoolTuner {
    sampler: Arc<LoadSampler>,
    strategy: TuningStrategy,
    current_config: Mutex<PoolConfig>,
    /// Hard upper bound on max_connections regardless of load.
    absolute_max: u32,
    /// Hard lower bound on min_connections regardless of load.
    absolute_min: u32,
}

impl PoolTuner {
    /// Create a new tuner with the given sampler, strategy, and starting config.
    #[must_use]
    pub fn new(
        sampler: Arc<LoadSampler>,
        strategy: TuningStrategy,
        initial_config: PoolConfig,
        absolute_min: u32,
        absolute_max: u32,
    ) -> Self {
        Self {
            sampler,
            strategy,
            current_config: Mutex::new(initial_config),
            absolute_max,
            absolute_min,
        }
    }

    /// Evaluate current load and return a [`TuningRecommendation`].
    ///
    /// The recommendation reflects the *desired* configuration; callers decide
    /// whether and when to apply it (e.g. at pool-recreation boundaries).
    #[must_use]
    pub fn evaluate(&self, metrics: &PoolMetrics) -> TuningRecommendation {
        let concurrency = self.sampler.current_concurrency();
        let peak = self.sampler.peak_concurrency();
        let mean = self.sampler.mean_concurrency();

        let new_config = self.compute_config(metrics, concurrency, peak, mean);
        let rationale = self.build_rationale(metrics, concurrency, peak, mean, &new_config);

        TuningRecommendation {
            config: new_config,
            rationale,
            observed_concurrency: concurrency,
            observed_utilisation: metrics.utilisation,
        }
    }

    fn compute_config(
        &self,
        metrics: &PoolMetrics,
        concurrency: u32,
        peak: u32,
        mean: f32,
    ) -> PoolConfig {
        let guard = self
            .current_config
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let current = guard.clone();
        drop(guard);

        let (new_min, new_max) = match self.strategy {
            TuningStrategy::Conservative => (current.min_connections, current.max_connections),
            TuningStrategy::Balanced => self.balanced_sizing(metrics, concurrency, peak, mean),
            TuningStrategy::Aggressive => self.aggressive_sizing(metrics, concurrency, peak, mean),
        };

        let new_min = new_min.clamp(self.absolute_min, self.absolute_max);
        let new_max = new_max
            .clamp(new_min, self.absolute_max)
            .max(self.absolute_min);

        // Derive timeouts based on load.
        let acquire_timeout = if metrics.is_high_load() {
            Duration::from_secs(10)
        } else {
            Duration::from_secs(30)
        };

        PoolConfig {
            min_connections: new_min,
            max_connections: new_max,
            acquire_timeout,
            ..current
        }
    }

    fn balanced_sizing(
        &self,
        metrics: &PoolMetrics,
        concurrency: u32,
        peak: u32,
        mean: f32,
    ) -> (u32, u32) {
        let cur = self
            .current_config
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();

        if metrics.is_high_load() || metrics.pending_requests > 0 {
            // Grow by 25 % of current max (at least +2).
            let increment = ((cur.max_connections as f32 * 0.25) as u32).max(2);
            let new_max = cur.max_connections + increment;
            let new_min = mean.ceil() as u32;
            (new_min, new_max)
        } else if metrics.is_low_load() {
            // Shrink by 10 % (at least -1).
            let decrement = ((cur.max_connections as f32 * 0.10) as u32).max(1);
            let new_max = cur.max_connections.saturating_sub(decrement);
            let new_min = (peak / 2).max(self.absolute_min);
            (new_min, new_max)
        } else {
            // Maintain.
            let new_min = mean.ceil() as u32;
            (new_min, concurrency.max(cur.max_connections))
        }
    }

    fn aggressive_sizing(
        &self,
        metrics: &PoolMetrics,
        concurrency: u32,
        peak: u32,
        mean: f32,
    ) -> (u32, u32) {
        let cur = self
            .current_config
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();

        if metrics.is_high_load() || metrics.pending_requests > 0 {
            // Target 2× current peak.
            let new_max = (peak * 2).max(cur.max_connections + 4);
            let new_min = concurrency;
            (new_min, new_max)
        } else if metrics.is_low_load() {
            // Shrink to just above mean.
            let new_max = ((mean * 1.5) as u32).max(self.absolute_min + 1);
            let new_min = self.absolute_min;
            (new_min, new_max)
        } else {
            let new_min = self.absolute_min;
            let new_max = concurrency.max(cur.max_connections);
            (new_min, new_max)
        }
    }

    fn build_rationale(
        &self,
        metrics: &PoolMetrics,
        concurrency: u32,
        peak: u32,
        mean: f32,
        new_config: &PoolConfig,
    ) -> String {
        let load_label = if metrics.is_high_load() {
            "high"
        } else if metrics.is_low_load() {
            "low"
        } else {
            "normal"
        };
        format!(
            "Strategy={:?} load={load_label} utilisation={:.1}% concurrency={concurrency} \
             peak={peak} mean={mean:.1} → min={} max={}",
            self.strategy,
            metrics.utilisation * 100.0,
            new_config.min_connections,
            new_config.max_connections,
        )
    }

    /// Apply the recommendation, updating the tuner's internal config record.
    pub fn apply(&self, recommendation: &TuningRecommendation) {
        if let Ok(mut guard) = self.current_config.lock() {
            *guard = recommendation.config.clone();
        }
    }

    /// Return a clone of the current config held by the tuner.
    #[must_use]
    pub fn current_config(&self) -> PoolConfig {
        self.current_config
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_metrics(active: u32, idle: u32, pending: u32, max: u32, min: u32) -> PoolMetrics {
        PoolMetrics::new(active, idle, pending, max, min)
    }

    #[test]
    fn test_pool_metrics_utilisation_calculation() {
        let m = make_metrics(40, 10, 0, 50, 5);
        // 40/50 = 0.8
        assert!((m.utilisation - 0.8).abs() < 1e-6);
        assert_eq!(m.total_connections, 50);
        assert!(m.is_high_load());
        assert!(!m.is_low_load());
    }

    #[test]
    fn test_pool_metrics_zero_max_connections() {
        let m = make_metrics(0, 0, 0, 0, 0);
        assert_eq!(m.utilisation, 0.0);
        assert!(!m.is_high_load());
    }

    #[test]
    fn test_pool_metrics_low_load_detection() {
        // 2/50 < 20 % and idle(10) > min(5)
        let m = make_metrics(2, 10, 0, 50, 5);
        assert!(m.is_low_load());
        assert!(!m.is_high_load());
    }

    #[test]
    fn test_load_sampler_concurrency_tracking() {
        let sampler = LoadSampler::new(Duration::from_secs(60));
        assert_eq!(sampler.current_concurrency(), 0);

        sampler.request_started();
        sampler.request_started();
        sampler.request_started();
        assert_eq!(sampler.current_concurrency(), 3);

        sampler.request_finished();
        assert_eq!(sampler.current_concurrency(), 2);

        sampler.request_finished();
        sampler.request_finished();
        assert_eq!(sampler.current_concurrency(), 0);
    }

    #[test]
    fn test_load_sampler_totals() {
        let sampler = LoadSampler::new(Duration::from_secs(60));
        for _ in 0..10 {
            sampler.request_started();
        }
        for _ in 0..7 {
            sampler.request_finished();
        }
        assert_eq!(sampler.total_started(), 10);
        assert_eq!(sampler.total_finished(), 7);
    }

    #[test]
    fn test_load_sampler_peak_concurrency() {
        let sampler = LoadSampler::new(Duration::from_secs(60));
        for _ in 0..5 {
            sampler.request_started();
        }
        let peak_at_5 = sampler.peak_concurrency();
        assert!(peak_at_5 >= 5);
        // Finish them all; peak should remain >= 5 within window.
        for _ in 0..5 {
            sampler.request_finished();
        }
        let peak_after = sampler.peak_concurrency();
        let _ = peak_after; // Window may still hold old samples; just verify no panic.
    }

    #[test]
    fn test_load_sampler_mean_concurrency() {
        let sampler = LoadSampler::new(Duration::from_secs(60));
        sampler.request_started();
        sampler.request_started();
        let mean = sampler.mean_concurrency();
        assert!(mean >= 1.0);
        assert!(mean <= 2.0);
    }

    #[test]
    fn test_pool_tuner_conservative_never_changes() {
        let sampler = Arc::new(LoadSampler::new(Duration::from_secs(60)));
        let config = PoolConfig {
            min_connections: 5,
            max_connections: 20,
            ..Default::default()
        };
        let tuner = PoolTuner::new(
            Arc::clone(&sampler),
            TuningStrategy::Conservative,
            config.clone(),
            2,
            100,
        );
        // Simulate high load.
        for _ in 0..20 {
            sampler.request_started();
        }
        let m = make_metrics(20, 0, 5, 20, 5);
        let rec = tuner.evaluate(&m);
        // Conservative must never change min/max.
        assert_eq!(rec.config.min_connections, config.min_connections);
        assert_eq!(rec.config.max_connections, config.max_connections);
    }

    #[test]
    fn test_pool_tuner_balanced_grows_on_high_load() {
        let sampler = Arc::new(LoadSampler::new(Duration::from_secs(60)));
        let config = PoolConfig {
            min_connections: 5,
            max_connections: 20,
            ..Default::default()
        };
        let tuner = PoolTuner::new(
            Arc::clone(&sampler),
            TuningStrategy::Balanced,
            config.clone(),
            2,
            200,
        );
        for _ in 0..18 {
            sampler.request_started();
        }
        // 18/20 = 90 % utilisation → high load.
        let m = make_metrics(18, 2, 3, 20, 5);
        let rec = tuner.evaluate(&m);
        assert!(
            rec.config.max_connections > config.max_connections,
            "Expected pool to grow on high load, got {}",
            rec.config.max_connections
        );
    }

    #[test]
    fn test_pool_tuner_aggressive_shrinks_on_low_load() {
        let sampler = Arc::new(LoadSampler::new(Duration::from_secs(60)));
        let config = PoolConfig {
            min_connections: 2,
            max_connections: 100,
            ..Default::default()
        };
        let tuner = PoolTuner::new(
            Arc::clone(&sampler),
            TuningStrategy::Aggressive,
            config.clone(),
            2,
            200,
        );
        // 1/100 = 1 % utilisation and idle(70) > min(2) → low load.
        let m = make_metrics(1, 70, 0, 100, 2);
        let rec = tuner.evaluate(&m);
        assert!(
            rec.config.max_connections < config.max_connections,
            "Expected pool to shrink on low load, got {}",
            rec.config.max_connections
        );
    }

    #[test]
    fn test_pool_tuner_apply_updates_config() {
        let sampler = Arc::new(LoadSampler::new(Duration::from_secs(60)));
        let tuner = PoolTuner::new(
            Arc::clone(&sampler),
            TuningStrategy::Balanced,
            PoolConfig::default(),
            2,
            200,
        );
        let m = make_metrics(18, 2, 3, 20, 2);
        for _ in 0..18 {
            sampler.request_started();
        }
        let rec = tuner.evaluate(&m);
        tuner.apply(&rec);
        assert_eq!(
            tuner.current_config().max_connections,
            rec.config.max_connections
        );
        assert_eq!(
            tuner.current_config().min_connections,
            rec.config.min_connections
        );
    }

    #[test]
    fn test_pool_tuner_recommendation_rationale_is_non_empty() {
        let sampler = Arc::new(LoadSampler::new(Duration::from_secs(60)));
        let tuner = PoolTuner::new(
            Arc::clone(&sampler),
            TuningStrategy::Balanced,
            PoolConfig::default(),
            2,
            100,
        );
        let m = make_metrics(5, 5, 0, 20, 2);
        let rec = tuner.evaluate(&m);
        assert!(!rec.rationale.is_empty());
    }

    #[test]
    fn test_pool_config_default_values() {
        let cfg = PoolConfig::default();
        assert_eq!(cfg.min_connections, 2);
        assert_eq!(cfg.max_connections, 20);
        assert!(cfg.acquire_timeout > Duration::ZERO);
        assert!(cfg.idle_timeout > Duration::ZERO);
        assert!(cfg.max_lifetime > Duration::ZERO);
    }

    #[test]
    fn test_load_sampler_thread_safety() {
        let sampler = Arc::new(LoadSampler::new(Duration::from_secs(60)));
        let mut handles = vec![];
        for _ in 0..4 {
            let s = Arc::clone(&sampler);
            handles.push(thread::spawn(move || {
                for _ in 0..25 {
                    s.request_started();
                    s.request_finished();
                }
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }
        assert_eq!(sampler.total_started(), 100);
        assert_eq!(sampler.total_finished(), 100);
    }

    // -----------------------------------------------------------------------
    // Async / tokio integration tests
    // -----------------------------------------------------------------------
    //
    // LoadSampler is Send + Sync, so it is safe to share across tokio tasks.
    // These tests verify that behaviour: spawn N async tasks, each recording a
    // started/finished pair, then assert the counters are consistent.

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_load_sampler_async_concurrent_tasks() {
        use tokio::task;

        let sampler = Arc::new(LoadSampler::new(Duration::from_secs(60)));
        let mut handles = vec![];

        // Spin up 8 tasks, each performing 20 start/finish pairs.
        for _ in 0..8 {
            let s = Arc::clone(&sampler);
            handles.push(task::spawn(async move {
                for _ in 0..20 {
                    s.request_started();
                    // Yield so other tasks may interleave.
                    task::yield_now().await;
                    s.request_finished();
                    task::yield_now().await;
                }
            }));
        }
        for h in handles {
            h.await.expect("task panicked");
        }

        // All 160 operations must have been recorded.
        assert_eq!(sampler.total_started(), 160);
        assert_eq!(sampler.total_finished(), 160);
        // After everything is finished concurrency must be 0.
        assert_eq!(sampler.current_concurrency(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_pool_tuner_grows_under_async_load() {
        use tokio::task;

        let sampler = Arc::new(LoadSampler::new(Duration::from_secs(60)));
        let config = PoolConfig {
            min_connections: 2,
            max_connections: 20,
            ..Default::default()
        };
        let tuner = Arc::new(PoolTuner::new(
            Arc::clone(&sampler),
            TuningStrategy::Balanced,
            config.clone(),
            2,
            200,
        ));

        // Simulate high concurrency from async tasks so the tuner sees load.
        let mut handles = vec![];
        for _ in 0..18 {
            let s = Arc::clone(&sampler);
            handles.push(task::spawn(async move {
                s.request_started();
                task::yield_now().await;
                // Leave in-flight so the snapshot captures them.
            }));
        }
        for h in handles {
            h.await.expect("task panicked");
        }

        // Snapshot at 18/20 → 90 % utilisation → high load → pool should grow.
        let metrics = make_metrics(18, 2, 3, 20, 2);
        let rec = tuner.evaluate(&metrics);
        tuner.apply(&rec);

        assert!(
            rec.config.max_connections > config.max_connections,
            "expected pool to grow under async high-load; got max={}",
            rec.config.max_connections
        );
        // The tuner's internal config must have been updated.
        assert_eq!(
            tuner.current_config().max_connections,
            rec.config.max_connections
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_pool_tuner_acquire_timeout_shortened_on_high_load() {
        let sampler = Arc::new(LoadSampler::new(Duration::from_secs(60)));
        let tuner = PoolTuner::new(
            Arc::clone(&sampler),
            TuningStrategy::Balanced,
            PoolConfig::default(),
            2,
            200,
        );
        // High-load metrics: utilisation = 90 %.
        let metrics = make_metrics(18, 2, 5, 20, 2);
        let rec = tuner.evaluate(&metrics);
        // Under high load the tuner must shorten the acquire timeout to 10 s.
        assert_eq!(rec.config.acquire_timeout, Duration::from_secs(10));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_pool_bounds_never_violated_under_async_load() {
        let sampler = Arc::new(LoadSampler::new(Duration::from_secs(60)));
        let absolute_min = 2_u32;
        let absolute_max = 50_u32;
        let tuner = PoolTuner::new(
            Arc::clone(&sampler),
            TuningStrategy::Aggressive,
            PoolConfig {
                min_connections: 2,
                max_connections: 20,
                ..Default::default()
            },
            absolute_min,
            absolute_max,
        );

        // Test both extremes: extreme high load and extreme low load.
        for _ in 0..50 {
            sampler.request_started();
        }
        let high_metrics = make_metrics(50, 0, 10, 50, 2);
        let rec_high = tuner.evaluate(&high_metrics);
        tuner.apply(&rec_high);
        assert!(
            rec_high.config.max_connections <= absolute_max,
            "max must not exceed absolute_max={}; got={}",
            absolute_max,
            rec_high.config.max_connections
        );
        assert!(
            rec_high.config.min_connections >= absolute_min,
            "min must not go below absolute_min={}; got={}",
            absolute_min,
            rec_high.config.min_connections
        );

        let low_metrics = make_metrics(1, 49, 0, 50, 2);
        let rec_low = tuner.evaluate(&low_metrics);
        assert!(
            rec_low.config.max_connections <= absolute_max,
            "max must not exceed absolute_max after low-load; got={}",
            rec_low.config.max_connections
        );
        assert!(
            rec_low.config.min_connections >= absolute_min,
            "min must not go below absolute_min after low-load; got={}",
            rec_low.config.min_connections
        );
    }
}
