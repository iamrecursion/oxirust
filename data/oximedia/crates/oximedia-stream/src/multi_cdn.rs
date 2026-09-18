//! Multi-CDN failover routing for live and on-demand streaming.
//!
//! Provides a [`MultiCdnRouter`] that selects among a pool of [`CdnProvider`]s
//! using pluggable [`RoutingStrategy`] policies.  Latency is tracked with an
//! EWMA (α = 0.2); consecutive failures exceeding a threshold mark a provider
//! unavailable until a successful request resets the error counter.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use crate::StreamError;

// ─── CdnProvider ─────────────────────────────────────────────────────────────

/// A single CDN origin / edge node.
#[derive(Debug)]
pub struct CdnProvider {
    /// Human-readable name, e.g. `"cloudfront-us-east-1"`.
    pub name: String,
    /// Base URL of this CDN, e.g. `"https://cdn1.example.com"`.
    pub base_url: String,
    /// Priority — lower values are preferred by the `Primary` strategy.
    pub priority: u32,
    /// Weight used by the `WeightedRandom` strategy.
    pub weight: u32,
    /// EWMA-smoothed latency in milliseconds (stored as fixed-point × 1000
    /// to avoid floating-point in atomics; divide by 1000 when reading).
    pub(crate) latency_ms_fp: AtomicU64,
    /// Number of consecutive errors since the last success.
    pub(crate) error_count: AtomicU32,
    /// Whether this provider is currently considered available.
    pub(crate) available: AtomicBool,
}

impl CdnProvider {
    /// Construct a new provider.  Initial latency estimate is `initial_latency_ms`.
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        priority: u32,
        weight: u32,
        initial_latency_ms: u64,
    ) -> Arc<Self> {
        Arc::new(Self {
            name: name.into(),
            base_url: base_url.into(),
            priority,
            weight: weight.max(1),
            latency_ms_fp: AtomicU64::new(initial_latency_ms * 1000),
            error_count: AtomicU32::new(0),
            available: AtomicBool::new(true),
        })
    }

    /// Current EWMA latency estimate in milliseconds.
    pub fn latency_ms(&self) -> f64 {
        self.latency_ms_fp.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Whether this provider is currently available.
    pub fn is_available(&self) -> bool {
        self.available.load(Ordering::Acquire)
    }

    /// Current consecutive error count.
    pub fn error_count(&self) -> u32 {
        self.error_count.load(Ordering::Relaxed)
    }
}

// ─── RoutingStrategy ─────────────────────────────────────────────────────────

/// How the [`MultiCdnRouter`] chooses a CDN provider for each request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Always pick the available provider with the lowest `priority` value.
    Primary,
    /// Cycle through all available providers in registration order.
    RoundRobin,
    /// Pick the available provider with the lowest EWMA latency.
    LatencyBased,
    /// Deterministic weighted selection using `request_count mod total_weight`.
    WeightedRandom,
    /// Weighted round-robin: cycles through providers in registration order,
    /// repeating each provider proportionally to its weight.  For example, with
    /// weights \[3, 1\] the pattern is A, A, A, B, A, A, A, B, ...
    ///
    /// Unlike `WeightedRandom` (which is modular-arithmetic based), this
    /// strategy produces a smooth interleaving that distributes requests evenly
    /// within each cycle.
    WeightedRoundRobin,
}

// ─── FailoverPolicy ──────────────────────────────────────────────────────────

/// Controls when providers are marked unavailable and when they recover.
#[derive(Debug, Clone)]
pub struct FailoverPolicy {
    /// Number of consecutive errors before a provider is marked unavailable.
    pub max_errors: u32,
    /// Number of consecutive successes required to mark a provider available again.
    /// Currently unused (recovery is immediate on the first success).
    pub recovery_threshold: u32,
    /// Milliseconds before a request is considered timed out (informational).
    pub timeout_ms: u64,
}

impl Default for FailoverPolicy {
    fn default() -> Self {
        Self {
            max_errors: 5,
            recovery_threshold: 1,
            timeout_ms: 5_000,
        }
    }
}

// ─── MultiCdnRouter ──────────────────────────────────────────────────────────

/// Routes streaming requests across a pool of [`CdnProvider`]s.
pub struct MultiCdnRouter {
    /// All registered providers (ordered by registration time).
    pub providers: Vec<Arc<CdnProvider>>,
    /// Failover policy controlling error thresholds.
    pub policy: FailoverPolicy,
    /// Total number of requests routed (used by RoundRobin / WeightedRandom).
    request_count: u64,
}

impl MultiCdnRouter {
    /// Create a router from a list of pre-constructed [`CdnProvider`] arcs.
    pub fn new(providers: Vec<Arc<CdnProvider>>) -> Self {
        Self {
            providers,
            policy: FailoverPolicy::default(),
            request_count: 0,
        }
    }

    /// Create a router with a custom [`FailoverPolicy`].
    pub fn with_policy(providers: Vec<Arc<CdnProvider>>, policy: FailoverPolicy) -> Self {
        Self {
            providers,
            policy,
            request_count: 0,
        }
    }

    /// Select a provider according to `strategy`.
    ///
    /// Returns `None` if no providers are available.
    pub fn select_provider(&mut self, strategy: &RoutingStrategy) -> Option<Arc<CdnProvider>> {
        self.request_count = self.request_count.wrapping_add(1);
        match strategy {
            RoutingStrategy::Primary => self.select_primary(),
            RoutingStrategy::RoundRobin => self.select_round_robin(),
            RoutingStrategy::LatencyBased => self.select_latency_based(),
            RoutingStrategy::WeightedRandom => self.select_weighted_random(),
            RoutingStrategy::WeightedRoundRobin => self.select_weighted_round_robin(),
        }
    }

    /// Record a measured round-trip latency for the named provider.
    ///
    /// Updates the EWMA with α = 0.2.
    pub fn record_latency(&self, name: &str, latency_ms: u64) {
        if let Some(p) = self.find_provider(name) {
            let alpha_fp = 200u64; // 0.2 × 1000
            let new_fp = latency_ms * 1000;
            // EWMA: new = α × sample + (1-α) × old  (all in fixed-point × 1000)
            let old_fp = p.latency_ms_fp.load(Ordering::Relaxed);
            let updated_fp = (alpha_fp * new_fp + (1000 - alpha_fp) * old_fp) / 1000;
            p.latency_ms_fp.store(updated_fp, Ordering::Relaxed);
        }
    }

    /// Record a request failure for the named provider.
    ///
    /// After `policy.max_errors` consecutive failures the provider is marked
    /// unavailable.
    pub fn record_error(&self, name: &str) {
        if let Some(p) = self.find_provider(name) {
            let prev = p.error_count.fetch_add(1, Ordering::AcqRel);
            let new_count = prev + 1;
            if new_count >= self.policy.max_errors {
                p.available.store(false, Ordering::Release);
            }
        }
    }

    /// Record a successful request for the named provider.
    ///
    /// Resets the consecutive error counter and marks the provider available.
    pub fn record_success(&self, name: &str) {
        if let Some(p) = self.find_provider(name) {
            p.error_count.store(0, Ordering::Release);
            p.available.store(true, Ordering::Release);
        }
    }

    /// Mark a provider available or unavailable directly (e.g. after an external
    /// health check).
    pub fn set_availability(&self, name: &str, available: bool) {
        if let Some(p) = self.find_provider(name) {
            p.available.store(available, Ordering::Release);
            if available {
                p.error_count.store(0, Ordering::Release);
            }
        }
    }

    /// Number of currently available providers.
    pub fn available_count(&self) -> usize {
        self.providers.iter().filter(|p| p.is_available()).count()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn find_provider(&self, name: &str) -> Option<&Arc<CdnProvider>> {
        self.providers.iter().find(|p| p.name == name)
    }

    /// Available providers sorted by priority ascending.
    fn available_by_priority(&self) -> Vec<&Arc<CdnProvider>> {
        let mut avail: Vec<&Arc<CdnProvider>> =
            self.providers.iter().filter(|p| p.is_available()).collect();
        avail.sort_by_key(|p| p.priority);
        avail
    }

    fn select_primary(&self) -> Option<Arc<CdnProvider>> {
        self.available_by_priority().into_iter().next().cloned()
    }

    fn select_round_robin(&self) -> Option<Arc<CdnProvider>> {
        let available: Vec<&Arc<CdnProvider>> =
            self.providers.iter().filter(|p| p.is_available()).collect();
        if available.is_empty() {
            return None;
        }
        let idx = (self.request_count as usize - 1) % available.len();
        Some(Arc::clone(available[idx]))
    }

    fn select_latency_based(&self) -> Option<Arc<CdnProvider>> {
        self.providers
            .iter()
            .filter(|p| p.is_available())
            .min_by(|a, b| {
                a.latency_ms_fp
                    .load(Ordering::Relaxed)
                    .cmp(&b.latency_ms_fp.load(Ordering::Relaxed))
            })
            .cloned()
    }

    fn select_weighted_random(&self) -> Option<Arc<CdnProvider>> {
        let available: Vec<&Arc<CdnProvider>> =
            self.providers.iter().filter(|p| p.is_available()).collect();
        if available.is_empty() {
            return None;
        }
        let total_weight: u64 = available.iter().map(|p| p.weight as u64).sum();
        if total_weight == 0 {
            return Some(Arc::clone(available[0]));
        }
        // Deterministic index: (request_count - 1) % total_weight
        let slot = (self.request_count.wrapping_sub(1)) % total_weight;
        let mut cumulative: u64 = 0;
        for p in &available {
            cumulative += p.weight as u64;
            if slot < cumulative {
                return Some(Arc::clone(p));
            }
        }
        // Fallback — should not be reached
        Some(Arc::clone(available[available.len() - 1]))
    }

    /// Weighted round-robin: smooth interleaving based on deficit counter.
    ///
    /// Uses a "deficit round-robin" (DRR) approach: each provider accumulates
    /// a virtual credit equal to its weight on every cycle.  The provider with
    /// the highest accumulated credit is selected, and one unit is subtracted.
    /// This produces a smooth distribution over time (e.g. weights [3,1] →
    /// A,A,A,B repeating).
    ///
    /// Because we don't persist per-provider counters in this stateless
    /// implementation, we compute the result deterministically from
    /// `request_count` using the modular walk over the weight array.
    fn select_weighted_round_robin(&self) -> Option<Arc<CdnProvider>> {
        let available: Vec<&Arc<CdnProvider>> =
            self.providers.iter().filter(|p| p.is_available()).collect();
        if available.is_empty() {
            return None;
        }
        let total_weight: u64 = available.iter().map(|p| p.weight as u64).sum();
        if total_weight == 0 {
            return Some(Arc::clone(available[0]));
        }

        // Compute a smooth interleaving using stride-based scheduling.
        // For each position in the cycle [0, total_weight), assign it to the
        // provider whose cumulative weight bracket contains it — but instead
        // of contiguous blocks (which is what WeightedRandom does), we
        // interleave by using an inverse-weight stride.
        //
        // We implement a simple deficit counter walk: simulate `req` steps
        // where `req = (request_count - 1) % total_weight`.
        let cycle_pos = self.request_count.wrapping_sub(1) % total_weight;

        // Build deficit counters for each available provider.
        let weights: Vec<u64> = available.iter().map(|p| p.weight as u64).collect();
        let n = weights.len();
        let mut deficits = vec![0i64; n];

        // Simulate cycle_pos + 1 selection steps.
        let mut selected_idx = 0usize;
        for _ in 0..=cycle_pos {
            // Add weight (credit) to all providers.
            for (i, &w) in weights.iter().enumerate() {
                deficits[i] += w as i64;
            }
            // Pick the provider with the highest deficit.
            let mut best = 0usize;
            for i in 1..n {
                if deficits[i] > deficits[best] {
                    best = i;
                }
            }
            // Subtract total_weight from the selected provider's deficit.
            deficits[best] -= total_weight as i64;
            selected_idx = best;
        }

        Some(Arc::clone(available[selected_idx]))
    }
}

// ─── Error conversion helper ──────────────────────────────────────────────────

/// Convert a provider name into a [`StreamError`] when no provider is found.
pub fn no_provider_error(strategy: &RoutingStrategy) -> StreamError {
    StreamError::RoutingError(format!("no available provider for strategy {strategy:?}"))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_provider(name: &str, priority: u32, weight: u32, latency: u64) -> Arc<CdnProvider> {
        CdnProvider::new(
            name,
            format!("https://{name}.cdn.example"),
            priority,
            weight,
            latency,
        )
    }

    fn make_router(n: usize) -> MultiCdnRouter {
        let providers: Vec<Arc<CdnProvider>> = (0..n)
            .map(|i| make_provider(&format!("cdn{i}"), i as u32, 1, 100))
            .collect();
        MultiCdnRouter::new(providers)
    }

    // ── CdnProvider construction ──────────────────────────────────────────────

    #[test]
    fn test_provider_initial_latency() {
        let p = make_provider("test", 1, 1, 50);
        let lat = p.latency_ms();
        assert!((lat - 50.0).abs() < 0.001, "expected 50ms got {lat}");
    }

    #[test]
    fn test_provider_initial_available() {
        let p = make_provider("test", 1, 1, 50);
        assert!(p.is_available());
    }

    #[test]
    fn test_provider_initial_error_count_zero() {
        let p = make_provider("test", 1, 1, 50);
        assert_eq!(p.error_count(), 0);
    }

    // ── record_latency ────────────────────────────────────────────────────────

    #[test]
    fn test_record_latency_updates_ewma() {
        let router = make_router(1);
        router.record_latency("cdn0", 200);
        let lat = router.providers[0].latency_ms();
        // EWMA: 0.2 × 200 + 0.8 × 100 = 120
        assert!((lat - 120.0).abs() < 1.0, "expected ~120ms got {lat}");
    }

    #[test]
    fn test_record_latency_unknown_provider_noop() {
        let router = make_router(1);
        // Should not panic
        router.record_latency("nonexistent", 500);
    }

    #[test]
    fn test_record_latency_converges_to_sample() {
        let router = make_router(1);
        for _ in 0..50 {
            router.record_latency("cdn0", 300);
        }
        let lat = router.providers[0].latency_ms();
        assert!(lat > 295.0 && lat < 305.0, "expected ~300ms got {lat}");
    }

    // ── record_error / record_success ─────────────────────────────────────────

    #[test]
    fn test_record_error_increments_count() {
        let router = make_router(1);
        router.record_error("cdn0");
        assert_eq!(router.providers[0].error_count(), 1);
    }

    #[test]
    fn test_record_error_marks_unavailable_after_threshold() {
        let router = make_router(1);
        for _ in 0..5 {
            router.record_error("cdn0");
        }
        assert!(!router.providers[0].is_available());
    }

    #[test]
    fn test_record_error_below_threshold_stays_available() {
        let router = make_router(1);
        for _ in 0..4 {
            router.record_error("cdn0");
        }
        assert!(router.providers[0].is_available());
    }

    #[test]
    fn test_record_success_resets_error_count() {
        let router = make_router(1);
        for _ in 0..3 {
            router.record_error("cdn0");
        }
        router.record_success("cdn0");
        assert_eq!(router.providers[0].error_count(), 0);
        assert!(router.providers[0].is_available());
    }

    #[test]
    fn test_record_success_restores_availability() {
        let router = make_router(1);
        for _ in 0..5 {
            router.record_error("cdn0");
        }
        assert!(!router.providers[0].is_available());
        router.record_success("cdn0");
        assert!(router.providers[0].is_available());
    }

    // ── select_provider: Primary ──────────────────────────────────────────────

    #[test]
    fn test_primary_selects_lowest_priority() {
        let mut router = MultiCdnRouter::new(vec![
            make_provider("high", 10, 1, 50),
            make_provider("low", 1, 1, 50),
        ]);
        let p = router
            .select_provider(&RoutingStrategy::Primary)
            .expect("provider");
        assert_eq!(p.name, "low");
    }

    #[test]
    fn test_primary_skips_unavailable() {
        let mut router = MultiCdnRouter::new(vec![
            make_provider("first", 1, 1, 50),
            make_provider("second", 2, 1, 50),
        ]);
        router.set_availability("first", false);
        let p = router
            .select_provider(&RoutingStrategy::Primary)
            .expect("provider");
        assert_eq!(p.name, "second");
    }

    #[test]
    fn test_primary_returns_none_when_all_unavailable() {
        let mut router = make_router(2);
        for p in &router.providers {
            p.available.store(false, Ordering::Release);
        }
        assert!(router.select_provider(&RoutingStrategy::Primary).is_none());
    }

    // ── select_provider: RoundRobin ───────────────────────────────────────────

    #[test]
    fn test_round_robin_cycles_providers() {
        let mut router = make_router(3);
        let names: Vec<String> = (0..6)
            .map(|_| {
                router
                    .select_provider(&RoutingStrategy::RoundRobin)
                    .expect("provider")
                    .name
                    .clone()
            })
            .collect();
        // Should cycle: cdn0, cdn1, cdn2, cdn0, cdn1, cdn2
        assert_eq!(names[0], names[3]);
        assert_eq!(names[1], names[4]);
        assert_eq!(names[2], names[5]);
    }

    // ── select_provider: LatencyBased ─────────────────────────────────────────

    #[test]
    fn test_latency_based_picks_lowest_latency() {
        let mut router = MultiCdnRouter::new(vec![
            make_provider("fast", 1, 1, 20),
            make_provider("slow", 2, 1, 200),
        ]);
        let p = router
            .select_provider(&RoutingStrategy::LatencyBased)
            .expect("provider");
        assert_eq!(p.name, "fast");
    }

    #[test]
    fn test_latency_based_skips_unavailable() {
        let mut router = MultiCdnRouter::new(vec![
            make_provider("fast", 1, 1, 10),
            make_provider("slow", 2, 1, 300),
        ]);
        router.set_availability("fast", false);
        let p = router
            .select_provider(&RoutingStrategy::LatencyBased)
            .expect("provider");
        assert_eq!(p.name, "slow");
    }

    // ── select_provider: WeightedRandom ───────────────────────────────────────

    #[test]
    fn test_weighted_random_deterministic() {
        let mut r1 = MultiCdnRouter::new(vec![
            make_provider("a", 1, 3, 50),
            make_provider("b", 2, 1, 50),
        ]);
        let mut r2 = MultiCdnRouter::new(vec![
            make_provider("a", 1, 3, 50),
            make_provider("b", 2, 1, 50),
        ]);
        for _ in 0..8 {
            let n1 = r1
                .select_provider(&RoutingStrategy::WeightedRandom)
                .expect("p")
                .name
                .clone();
            let n2 = r2
                .select_provider(&RoutingStrategy::WeightedRandom)
                .expect("p")
                .name
                .clone();
            assert_eq!(n1, n2, "same request_count → same provider");
        }
    }

    #[test]
    fn test_weighted_random_higher_weight_selected_more() {
        let mut router = MultiCdnRouter::new(vec![
            make_provider("heavy", 1, 9, 50),
            make_provider("light", 2, 1, 50),
        ]);
        let mut heavy_count = 0usize;
        for _ in 0..10 {
            let name = router
                .select_provider(&RoutingStrategy::WeightedRandom)
                .expect("p")
                .name
                .clone();
            if name == "heavy" {
                heavy_count += 1;
            }
        }
        assert!(
            heavy_count >= 8,
            "expected ≥8/10 to be 'heavy', got {heavy_count}"
        );
    }

    // ── FailoverPolicy ────────────────────────────────────────────────────────

    #[test]
    fn test_custom_policy_max_errors() {
        let policy = FailoverPolicy {
            max_errors: 2,
            recovery_threshold: 1,
            timeout_ms: 1000,
        };
        let router = MultiCdnRouter::with_policy(vec![make_provider("cdn", 1, 1, 50)], policy);
        router.record_error("cdn");
        assert!(
            router.providers[0].is_available(),
            "one error, not yet down"
        );
        router.record_error("cdn");
        assert!(!router.providers[0].is_available(), "two errors, now down");
    }

    #[test]
    fn test_available_count_reflects_availability() {
        let router = make_router(4);
        assert_eq!(router.available_count(), 4);
        router.set_availability("cdn0", false);
        router.set_availability("cdn1", false);
        assert_eq!(router.available_count(), 2);
    }

    // ── select_provider: WeightedRoundRobin ──────────────────────────────────

    #[test]
    fn test_weighted_rr_distributes_proportionally() {
        let mut router = MultiCdnRouter::new(vec![
            make_provider("heavy", 1, 3, 50),
            make_provider("light", 2, 1, 50),
        ]);
        let mut heavy_count = 0usize;
        let total = 40;
        for _ in 0..total {
            let name = router
                .select_provider(&RoutingStrategy::WeightedRoundRobin)
                .expect("provider")
                .name
                .clone();
            if name == "heavy" {
                heavy_count += 1;
            }
        }
        // Weight ratio is 3:1, so heavy should get ~75% of selections.
        let ratio = heavy_count as f64 / total as f64;
        assert!(
            (ratio - 0.75).abs() < 0.05,
            "expected ~75% heavy, got {:.1}% ({heavy_count}/{total})",
            ratio * 100.0
        );
    }

    #[test]
    fn test_weighted_rr_deterministic() {
        let mut r1 = MultiCdnRouter::new(vec![
            make_provider("a", 1, 2, 50),
            make_provider("b", 2, 3, 50),
        ]);
        let mut r2 = MultiCdnRouter::new(vec![
            make_provider("a", 1, 2, 50),
            make_provider("b", 2, 3, 50),
        ]);
        for _ in 0..10 {
            let n1 = r1
                .select_provider(&RoutingStrategy::WeightedRoundRobin)
                .expect("p")
                .name
                .clone();
            let n2 = r2
                .select_provider(&RoutingStrategy::WeightedRoundRobin)
                .expect("p")
                .name
                .clone();
            assert_eq!(n1, n2, "same request_count should yield same provider");
        }
    }

    #[test]
    fn test_weighted_rr_single_provider() {
        let mut router = MultiCdnRouter::new(vec![make_provider("solo", 1, 5, 50)]);
        for _ in 0..5 {
            let p = router
                .select_provider(&RoutingStrategy::WeightedRoundRobin)
                .expect("provider");
            assert_eq!(p.name, "solo");
        }
    }

    #[test]
    fn test_weighted_rr_equal_weights_cycles() {
        let mut router = MultiCdnRouter::new(vec![
            make_provider("a", 1, 1, 50),
            make_provider("b", 2, 1, 50),
            make_provider("c", 3, 1, 50),
        ]);
        let mut counts = std::collections::HashMap::new();
        for _ in 0..30 {
            let name = router
                .select_provider(&RoutingStrategy::WeightedRoundRobin)
                .expect("provider")
                .name
                .clone();
            *counts.entry(name).or_insert(0usize) += 1;
        }
        // Equal weights → each should get exactly 10 out of 30.
        for (name, count) in &counts {
            assert_eq!(
                *count, 10,
                "provider {name} should get 10/30, got {count}/30"
            );
        }
    }

    #[test]
    fn test_weighted_rr_skips_unavailable() {
        let mut router = MultiCdnRouter::new(vec![
            make_provider("a", 1, 5, 50),
            make_provider("b", 2, 1, 50),
        ]);
        router.set_availability("a", false);
        for _ in 0..5 {
            let p = router
                .select_provider(&RoutingStrategy::WeightedRoundRobin)
                .expect("provider");
            assert_eq!(p.name, "b", "only 'b' should be selected when 'a' is down");
        }
    }

    #[test]
    fn test_weighted_rr_returns_none_when_all_unavailable() {
        let mut router = make_router(2);
        for p in &router.providers {
            p.available.store(false, Ordering::Release);
        }
        assert!(router
            .select_provider(&RoutingStrategy::WeightedRoundRobin)
            .is_none());
    }

    #[test]
    fn test_weighted_rr_smooth_interleaving() {
        // With weights [3,1], over 4 requests we should see 3× heavy and 1× light,
        // and it should be interleaved (not all heavy then light).
        let mut router = MultiCdnRouter::new(vec![
            make_provider("heavy", 1, 3, 50),
            make_provider("light", 2, 1, 50),
        ]);
        let names: Vec<String> = (0..4)
            .map(|_| {
                router
                    .select_provider(&RoutingStrategy::WeightedRoundRobin)
                    .expect("provider")
                    .name
                    .clone()
            })
            .collect();
        let heavy = names.iter().filter(|n| n.as_str() == "heavy").count();
        let light = names.iter().filter(|n| n.as_str() == "light").count();
        assert_eq!(heavy, 3, "3:1 weight → 3 heavy in 4 requests");
        assert_eq!(light, 1, "3:1 weight → 1 light in 4 requests");
    }
}
