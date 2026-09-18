//! Multi-CDN coordination — manage traffic across multiple CDN providers,
//! implement provider failover, weighted traffic splitting, and health-aware
//! routing decisions.
//!
//! # Overview
//!
//! `MultiCdnRouter` maintains a registry of `CdnProvider`s and distributes
//! requests using one of four strategies:
//!
//! - `RoutingStrategy::Primary`       — always use the highest-priority healthy provider.
//! - `RoutingStrategy::WeightedSplit` — distribute traffic by normalized weight.
//! - `RoutingStrategy::LeastErrors`   — prefer the provider with the lowest recent error rate.
//! - `RoutingStrategy::LowestLatency` — prefer the provider with the lowest EWMA latency.
//!
//! Each provider has atomic health state and EWMA latency tracking analogous to
//! [`OriginServer`](crate::origin_failover::OriginServer), but at the CDN-provider
//! level.  No external HTTP calls are made — callers record outcomes via
//! `MultiCdnRouter::record_success` and `MultiCdnRouter::record_failure`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by multi-CDN routing operations.
#[derive(Debug, Error)]
pub enum MultiCdnError {
    /// No healthy CDN provider is available.
    #[error("no healthy CDN providers available")]
    NoHealthyProviders,
    /// A provider with the given ID was not found.
    #[error("CDN provider '{0}' not found")]
    NotFound(String),
}

// ─── RoutingStrategy ─────────────────────────────────────────────────────────

/// How [`MultiCdnRouter`] selects a provider for each request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Always route to the healthy provider with the lowest `priority` value.
    Primary,
    /// Distribute traffic proportionally to `weight`.
    WeightedSplit,
    /// Prefer the healthy provider with the lowest recent error rate.
    LeastErrors,
    /// Prefer the healthy provider with the lowest EWMA response latency.
    LowestLatency,
}

// ─── ProviderStatus ──────────────────────────────────────────────────────────

/// Coarse health status of a CDN provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderStatus {
    /// Fully operational.
    Healthy,
    /// Elevated error rate / degraded performance, but still serving.
    Degraded,
    /// Unavailable — no traffic should be sent.
    Down,
}

// ─── CdnProvider ─────────────────────────────────────────────────────────────

/// A single CDN provider entry managed by [`MultiCdnRouter`].
pub struct CdnProvider {
    /// Unique provider identifier (e.g. `"cloudflare"`, `"fastly"`).
    pub id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Base URL prefix used for traffic steering (informational).
    pub base_url: String,
    /// Traffic weight (higher = more traffic in [`RoutingStrategy::WeightedSplit`]).
    pub weight: u32,
    /// Lower value = higher priority for [`RoutingStrategy::Primary`].
    pub priority: u8,
    /// Whether the provider is currently healthy.
    healthy: AtomicBool,
    /// Consecutive failure counter.
    consecutive_failures: AtomicU32,
    /// Mark as unhealthy after this many consecutive failures.
    pub failure_threshold: u32,
    /// Consecutive successes needed to recover from unhealthy state.
    pub recovery_threshold: u32,
    consecutive_successes: AtomicU32,
    /// EWMA latency in milliseconds (α = 0.3), behind a parking_lot::Mutex.
    ewma_latency_ms: Mutex<f64>,
    /// Total successful requests.
    total_success: AtomicU64,
    /// Total failed requests.
    total_errors: AtomicU64,
    /// Sliding-window error samples for error-rate tracking.
    error_window: Mutex<VecDeque<(Instant, bool)>>,
    /// Window duration for error-rate calculation.
    error_window_dur: Duration,
}

impl std::fmt::Debug for CdnProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CdnProvider")
            .field("id", &self.id)
            .field("display_name", &self.display_name)
            .field("weight", &self.weight)
            .field("priority", &self.priority)
            .field("healthy", &self.healthy.load(Ordering::Relaxed))
            .finish()
    }
}

impl CdnProvider {
    /// Create a new provider entry.
    ///
    /// - `weight`   — traffic weight for [`RoutingStrategy::WeightedSplit`].
    /// - `priority` — priority for [`RoutingStrategy::Primary`] (0 = highest).
    pub fn new(id: &str, display_name: &str, base_url: &str, weight: u32, priority: u8) -> Self {
        Self {
            id: id.to_string(),
            display_name: display_name.to_string(),
            base_url: base_url.to_string(),
            weight,
            priority,
            healthy: AtomicBool::new(true),
            consecutive_failures: AtomicU32::new(0),
            failure_threshold: 3,
            recovery_threshold: 1,
            consecutive_successes: AtomicU32::new(0),
            ewma_latency_ms: Mutex::new(100.0),
            total_success: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            error_window: Mutex::new(VecDeque::new()),
            error_window_dur: Duration::from_secs(60),
        }
    }

    /// Returns `true` if the provider is currently healthy.
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    /// Current coarse health status.
    pub fn status(&self) -> ProviderStatus {
        if !self.is_healthy() {
            return ProviderStatus::Down;
        }
        let rate = self.error_rate();
        if rate > 0.1 {
            ProviderStatus::Degraded
        } else {
            ProviderStatus::Healthy
        }
    }

    /// Record a successful request with its latency.
    pub fn record_success(&self, latency_ms: f64) {
        self.total_success.fetch_add(1, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);
        let successes = self.consecutive_successes.fetch_add(1, Ordering::Relaxed) + 1;
        if successes >= self.recovery_threshold {
            self.healthy.store(true, Ordering::Release);
        }
        // Update EWMA — parking_lot::Mutex is infallible.
        let mut ewma = self.ewma_latency_ms.lock();
        *ewma = 0.3 * latency_ms + 0.7 * (*ewma);
        drop(ewma);
        // Update error window.
        self.push_error_window(false);
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
        self.consecutive_successes.store(0, Ordering::Relaxed);
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        if failures >= self.failure_threshold {
            self.healthy.store(false, Ordering::Release);
        }
        self.push_error_window(true);
    }

    /// EWMA latency in milliseconds.
    pub fn ewma_latency_ms(&self) -> f64 {
        *self.ewma_latency_ms.lock()
    }

    /// Recent error rate over the sliding window (0.0–1.0).
    pub fn error_rate(&self) -> f64 {
        let mut window = self.error_window.lock();
        self.evict_window(&mut window);
        if window.is_empty() {
            return 0.0;
        }
        let errors = window.iter().filter(|(_, failed)| *failed).count();
        errors as f64 / window.len() as f64
    }

    /// Total successful requests.
    pub fn total_success(&self) -> u64 {
        self.total_success.load(Ordering::Relaxed)
    }

    /// Total failed requests.
    pub fn total_errors(&self) -> u64 {
        self.total_errors.load(Ordering::Relaxed)
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    fn push_error_window(&self, failed: bool) {
        let mut window = self.error_window.lock();
        self.evict_window(&mut window);
        window.push_back((Instant::now(), failed));
    }

    fn evict_window(&self, window: &mut VecDeque<(Instant, bool)>) {
        let cutoff = Instant::now()
            .checked_sub(self.error_window_dur)
            .unwrap_or_else(Instant::now);
        while let Some((ts, _)) = window.front() {
            if *ts < cutoff {
                window.pop_front();
            } else {
                break;
            }
        }
    }
}

// ─── SplitWeightState ────────────────────────────────────────────────────────

/// Internal round-robin cursor for weighted splits.
struct WeightedSplitState {
    index: usize,
}

// ─── MultiCdnRouter ──────────────────────────────────────────────────────────

/// Routes requests across multiple CDN providers.
pub struct MultiCdnRouter {
    providers: Vec<Arc<CdnProvider>>,
    strategy: RoutingStrategy,
    /// Internal cursor for weighted-split round-robin.
    split_state: Mutex<WeightedSplitState>,
}

impl MultiCdnRouter {
    /// Create a router with no providers and the given strategy.
    pub fn new(strategy: RoutingStrategy) -> Self {
        Self {
            providers: Vec::new(),
            strategy,
            split_state: Mutex::new(WeightedSplitState { index: 0 }),
        }
    }

    /// Register a CDN provider.
    pub fn add_provider(&mut self, provider: Arc<CdnProvider>) {
        self.providers.push(provider);
    }

    /// Register a CDN provider by value.
    pub fn add_provider_owned(&mut self, provider: CdnProvider) {
        self.providers.push(Arc::new(provider));
    }

    /// Select the best provider for the next request.
    ///
    /// Returns `Err(MultiCdnError::NoHealthyProviders)` when all providers are
    /// down.
    pub fn select(&self) -> Result<Arc<CdnProvider>, MultiCdnError> {
        match &self.strategy {
            RoutingStrategy::Primary => self.select_primary(),
            RoutingStrategy::WeightedSplit => self.select_weighted(),
            RoutingStrategy::LeastErrors => self.select_least_errors(),
            RoutingStrategy::LowestLatency => self.select_lowest_latency(),
        }
    }

    /// Record a successful request outcome for provider `id`.
    pub fn record_success(&self, id: &str, latency_ms: f64) -> Result<(), MultiCdnError> {
        self.find(id)?.record_success(latency_ms);
        Ok(())
    }

    /// Record a failed request outcome for provider `id`.
    pub fn record_failure(&self, id: &str) -> Result<(), MultiCdnError> {
        self.find(id)?.record_failure();
        Ok(())
    }

    /// Return a snapshot of all providers and their statuses.
    pub fn provider_statuses(&self) -> Vec<(String, ProviderStatus)> {
        self.providers
            .iter()
            .map(|p| (p.id.clone(), p.status()))
            .collect()
    }

    /// Number of healthy providers.
    pub fn healthy_count(&self) -> usize {
        self.providers.iter().filter(|p| p.is_healthy()).count()
    }

    /// All provider handles.
    pub fn providers(&self) -> &[Arc<CdnProvider>] {
        &self.providers
    }

    /// Find a provider by ID.
    pub fn get_provider(&self, id: &str) -> Option<Arc<CdnProvider>> {
        self.providers.iter().find(|p| p.id == id).cloned()
    }

    // ── Selection strategies ──────────────────────────────────────────────

    fn select_primary(&self) -> Result<Arc<CdnProvider>, MultiCdnError> {
        self.providers
            .iter()
            .filter(|p| p.is_healthy())
            .min_by_key(|p| p.priority)
            .cloned()
            .ok_or(MultiCdnError::NoHealthyProviders)
    }

    fn select_weighted(&self) -> Result<Arc<CdnProvider>, MultiCdnError> {
        let healthy: Vec<&Arc<CdnProvider>> =
            self.providers.iter().filter(|p| p.is_healthy()).collect();
        if healthy.is_empty() {
            return Err(MultiCdnError::NoHealthyProviders);
        }
        let total: u32 = healthy.iter().map(|p| p.weight).sum();
        if total == 0 {
            return Err(MultiCdnError::NoHealthyProviders);
        }
        let mut state = self.split_state.lock();
        state.index = (state.index + 1) % total as usize;
        let target = state.index as u32;
        drop(state);
        let mut cum = 0u32;
        for p in &healthy {
            cum += p.weight;
            if target < cum {
                return Ok(Arc::clone(p));
            }
        }
        healthy
            .last()
            .map(|p| Arc::clone(p))
            .ok_or(MultiCdnError::NoHealthyProviders)
    }

    fn select_least_errors(&self) -> Result<Arc<CdnProvider>, MultiCdnError> {
        self.providers
            .iter()
            .filter(|p| p.is_healthy())
            .min_by(|a, b| {
                a.error_rate()
                    .partial_cmp(&b.error_rate())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or(MultiCdnError::NoHealthyProviders)
    }

    fn select_lowest_latency(&self) -> Result<Arc<CdnProvider>, MultiCdnError> {
        self.providers
            .iter()
            .filter(|p| p.is_healthy())
            .min_by(|a, b| {
                a.ewma_latency_ms()
                    .partial_cmp(&b.ewma_latency_ms())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or(MultiCdnError::NoHealthyProviders)
    }

    fn find(&self, id: &str) -> Result<&Arc<CdnProvider>, MultiCdnError> {
        self.providers
            .iter()
            .find(|p| p.id == id)
            .ok_or_else(|| MultiCdnError::NotFound(id.to_string()))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_provider(id: &str, weight: u32, priority: u8) -> Arc<CdnProvider> {
        Arc::new(CdnProvider::new(
            id,
            id,
            &format!("https://{id}.cdn.example.com"),
            weight,
            priority,
        ))
    }

    // 1. Primary strategy selects lowest priority
    #[test]
    fn test_primary_selects_lowest_priority() {
        let mut router = MultiCdnRouter::new(RoutingStrategy::Primary);
        router.add_provider(make_provider("cf", 1, 0)); // primary
        router.add_provider(make_provider("fastly", 1, 1)); // secondary
        let sel = router.select().expect("provider");
        assert_eq!(sel.id, "cf");
    }

    // 2. Primary skips unhealthy providers
    #[test]
    fn test_primary_skips_unhealthy() {
        let mut router = MultiCdnRouter::new(RoutingStrategy::Primary);
        let cf = make_provider("cf", 1, 0);
        cf.healthy.store(false, Ordering::Relaxed);
        router.add_provider(cf);
        router.add_provider(make_provider("fastly", 1, 1));
        let sel = router.select().expect("provider");
        assert_eq!(sel.id, "fastly");
    }

    // 3. No healthy providers → error
    #[test]
    fn test_no_healthy_providers() {
        let mut router = MultiCdnRouter::new(RoutingStrategy::Primary);
        let p = make_provider("p", 1, 0);
        p.healthy.store(false, Ordering::Relaxed);
        router.add_provider(p);
        assert!(matches!(
            router.select(),
            Err(MultiCdnError::NoHealthyProviders)
        ));
    }

    // 4. WeightedSplit distributes traffic
    #[test]
    fn test_weighted_split_distributes() {
        let mut router = MultiCdnRouter::new(RoutingStrategy::WeightedSplit);
        router.add_provider(make_provider("a", 1, 0));
        router.add_provider(make_provider("b", 1, 0));
        let mut seen_a = false;
        let mut seen_b = false;
        for _ in 0..20 {
            let sel = router.select().expect("provider");
            match sel.id.as_str() {
                "a" => seen_a = true,
                "b" => seen_b = true,
                _ => {}
            }
        }
        assert!(seen_a && seen_b);
    }

    // 5. LeastErrors selects lowest error rate
    #[test]
    fn test_least_errors_selects_lowest() {
        let mut router = MultiCdnRouter::new(RoutingStrategy::LeastErrors);
        let good = make_provider("good", 1, 0);
        let bad = make_provider("bad", 1, 0);
        // Drive bad's error rate up.
        for _ in 0..5 {
            bad.record_failure();
        }
        bad.healthy.store(true, Ordering::Release); // keep healthy for comparison
        router.add_provider(Arc::clone(&bad));
        router.add_provider(Arc::clone(&good));
        let sel = router.select().expect("provider");
        assert_eq!(sel.id, "good");
    }

    // 6. LowestLatency selects fastest provider
    #[test]
    fn test_lowest_latency_selects_fastest() {
        let mut router = MultiCdnRouter::new(RoutingStrategy::LowestLatency);
        let fast = make_provider("fast", 1, 0);
        let slow = make_provider("slow", 1, 0);
        for _ in 0..5 {
            fast.record_success(10.0);
            slow.record_success(500.0);
        }
        router.add_provider(Arc::clone(&slow));
        router.add_provider(Arc::clone(&fast));
        let sel = router.select().expect("provider");
        assert_eq!(sel.id, "fast");
    }

    // 7. record_success updates EWMA
    #[test]
    fn test_record_success_updates_ewma() {
        let p = make_provider("p", 1, 0);
        p.record_success(10.0); // 0.3*10 + 0.7*100 = 73
        let ewma = p.ewma_latency_ms();
        assert!((ewma - 73.0).abs() < 1e-4, "ewma={ewma}");
    }

    // 8. record_failure marks down after threshold
    #[test]
    fn test_record_failure_marks_down() {
        let p = make_provider("p", 1, 0);
        p.record_failure();
        p.record_failure();
        assert!(p.is_healthy());
        p.record_failure(); // 3rd → down
        assert!(!p.is_healthy());
    }

    // 9. record_success recovers provider
    #[test]
    fn test_record_success_recovers() {
        let p = make_provider("p", 1, 0);
        p.healthy.store(false, Ordering::Relaxed);
        p.record_success(50.0);
        assert!(p.is_healthy());
    }

    // 10. error_rate calculation
    #[test]
    fn test_error_rate() {
        let p = make_provider("p", 1, 0);
        for _ in 0..3 {
            p.record_failure();
        }
        p.healthy.store(true, Ordering::Release);
        for _ in 0..7 {
            p.record_success(50.0);
        }
        let rate = p.error_rate();
        assert!((rate - 0.3).abs() < 1e-10, "rate={rate}");
    }

    // 11. provider_statuses snapshot
    #[test]
    fn test_provider_statuses() {
        let mut router = MultiCdnRouter::new(RoutingStrategy::Primary);
        router.add_provider(make_provider("cf", 1, 0));
        let statuses = router.provider_statuses();
        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0].0, "cf");
        assert_eq!(statuses[0].1, ProviderStatus::Healthy);
    }

    // 12. healthy_count
    #[test]
    fn test_healthy_count() {
        let mut router = MultiCdnRouter::new(RoutingStrategy::Primary);
        let p1 = make_provider("p1", 1, 0);
        let p2 = make_provider("p2", 1, 1);
        p2.healthy.store(false, Ordering::Relaxed);
        router.add_provider(p1);
        router.add_provider(p2);
        assert_eq!(router.healthy_count(), 1);
    }

    // 13. get_provider
    #[test]
    fn test_get_provider() {
        let mut router = MultiCdnRouter::new(RoutingStrategy::Primary);
        router.add_provider(make_provider("cf", 1, 0));
        assert!(router.get_provider("cf").is_some());
        assert!(router.get_provider("ghost").is_none());
    }

    // 14. record_success / record_failure via router
    #[test]
    fn test_router_record_outcomes() {
        let mut router = MultiCdnRouter::new(RoutingStrategy::Primary);
        router.add_provider(make_provider("cf", 1, 0));
        router.record_success("cf", 42.0).expect("ok");
        router.record_failure("cf").expect("ok");
        let p = router.get_provider("cf").expect("cf");
        assert_eq!(p.total_success(), 1);
        assert_eq!(p.total_errors(), 1);
    }

    // 15. record_failure on unknown provider
    #[test]
    fn test_record_outcome_unknown_provider() {
        let router = MultiCdnRouter::new(RoutingStrategy::Primary);
        assert!(matches!(
            router.record_failure("ghost"),
            Err(MultiCdnError::NotFound(_))
        ));
    }

    // 16. ProviderStatus Degraded when error rate > 10%
    #[test]
    fn test_provider_status_degraded() {
        let p = make_provider("p", 1, 0);
        // 2 failures out of 10 = 20% > 10%
        for _ in 0..2 {
            p.record_failure();
        }
        p.healthy.store(true, Ordering::Release);
        for _ in 0..8 {
            p.record_success(50.0);
        }
        let status = p.status();
        assert_eq!(status, ProviderStatus::Degraded);
    }

    // 17. WeightedSplit with unequal weights favours heavier provider
    #[test]
    fn test_weighted_split_unequal() {
        let mut router = MultiCdnRouter::new(RoutingStrategy::WeightedSplit);
        router.add_provider(make_provider("heavy", 3, 0));
        router.add_provider(make_provider("light", 1, 0));
        let mut heavy = 0u32;
        let mut light = 0u32;
        for _ in 0..100 {
            match router.select().expect("ok").id.as_str() {
                "heavy" => heavy += 1,
                "light" => light += 1,
                _ => {}
            }
        }
        assert!(heavy > light, "heavy={heavy} light={light}");
    }

    // 18. add_provider_owned
    #[test]
    fn test_add_provider_owned() {
        let mut router = MultiCdnRouter::new(RoutingStrategy::Primary);
        router.add_provider_owned(CdnProvider::new(
            "p",
            "P",
            "https://p.cdn.example.com",
            1,
            0,
        ));
        assert_eq!(router.providers().len(), 1);
    }

    // 19. total_success and total_errors counters
    #[test]
    fn test_total_counters() {
        let p = make_provider("p", 1, 0);
        p.record_success(10.0);
        p.record_success(20.0);
        p.record_failure();
        assert_eq!(p.total_success(), 2);
        assert_eq!(p.total_errors(), 1);
    }

    // 20. EWMA converges toward target latency
    #[test]
    fn test_ewma_converges() {
        let p = make_provider("p", 1, 0);
        for _ in 0..30 {
            p.record_success(10.0);
        }
        let ewma = p.ewma_latency_ms();
        assert!(ewma < 20.0, "ewma should converge near 10ms, got {ewma}");
    }
}
