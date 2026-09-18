//! Routing strategies for CDN provider selection.
//!
//! This module provides various routing strategies including round-robin,
//! weighted round-robin, least connections, least latency, geographic proximity,
//! and cost-based routing.

use super::health::HealthChecker;
use super::RequestContext;
use crate::error::{NetError, NetResult};
use parking_lot::RwLock;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Routing strategy for provider selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Round-robin: Distribute requests evenly across providers.
    RoundRobin,
    /// Weighted round-robin: Distribute based on provider weights.
    WeightedRoundRobin,
    /// Least connections: Select provider with fewest active connections.
    LeastConnections,
    /// Least latency: Select provider with lowest latency.
    LeastLatency,
    /// Geographic proximity: Select closest provider to client.
    Geographic,
    /// Cost-based: Select cheapest provider that meets SLA.
    CostBased,
    /// Priority-based: Select highest priority available provider.
    Priority,
    /// Random: Randomly select a provider.
    Random,
    /// Hash-based: Consistent hashing based on session/content.
    HashBased,
}

impl RoutingStrategy {
    /// Returns the strategy name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::RoundRobin => "Round Robin",
            Self::WeightedRoundRobin => "Weighted Round Robin",
            Self::LeastConnections => "Least Connections",
            Self::LeastLatency => "Least Latency",
            Self::Geographic => "Geographic",
            Self::CostBased => "Cost-Based",
            Self::Priority => "Priority",
            Self::Random => "Random",
            Self::HashBased => "Hash-Based",
        }
    }
}

/// Session affinity (sticky sessions) configuration.
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionAffinity {
    /// Enable session affinity.
    pub enabled: bool,
    /// Session timeout duration.
    pub timeout: Duration,
    /// Session-to-provider mappings.
    sessions: HashMap<String, SessionBinding>,
}

/// Session binding to a provider.
#[derive(Debug, Serialize, Deserialize)]
struct SessionBinding {
    /// Provider ID.
    provider_id: String,
    /// Binding creation time.
    created_at: SystemTime,
    /// Last access time.
    last_access: SystemTime,
}

impl SessionAffinity {
    /// Creates new session affinity manager.
    #[must_use]
    pub fn new(enabled: bool, timeout: Duration) -> Self {
        Self {
            enabled,
            timeout,
            sessions: HashMap::new(),
        }
    }

    /// Gets the bound provider for a session.
    pub fn get_provider(&mut self, session_id: &str) -> Option<String> {
        if !self.enabled {
            return None;
        }

        let binding = self.sessions.get_mut(session_id)?;

        // Check if session has expired
        if let Ok(elapsed) = binding.last_access.elapsed() {
            if elapsed > self.timeout {
                self.sessions.remove(session_id);
                return None;
            }
        }

        binding.last_access = SystemTime::now();
        Some(binding.provider_id.clone())
    }

    /// Binds a session to a provider.
    pub fn bind_session(&mut self, session_id: String, provider_id: String) {
        if !self.enabled {
            return;
        }

        let now = SystemTime::now();
        self.sessions.insert(
            session_id,
            SessionBinding {
                provider_id,
                created_at: now,
                last_access: now,
            },
        );
    }

    /// Removes expired sessions.
    pub fn cleanup_expired(&mut self) {
        if !self.enabled {
            return;
        }

        let timeout = self.timeout;
        self.sessions.retain(|_, binding| {
            binding
                .last_access
                .elapsed()
                .map_or(true, |elapsed| elapsed <= timeout)
        });
    }

    /// Unbinds a session.
    pub fn unbind_session(&mut self, session_id: &str) {
        self.sessions.remove(session_id);
    }

    /// Gets active session count.
    #[must_use]
    pub fn active_sessions(&self) -> usize {
        self.sessions.len()
    }
}

/// A/B testing configuration.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbTestConfig {
    /// Enable A/B testing.
    pub enabled: bool,
    /// Test groups.
    pub groups: Vec<AbTestGroup>,
}

/// A/B test group.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbTestGroup {
    /// Group name.
    pub name: String,
    /// Provider IDs in this group.
    pub provider_ids: Vec<String>,
    /// Traffic percentage (0-100).
    pub traffic_percentage: u8,
}

impl AbTestConfig {
    /// Creates a new A/B test configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled: false,
            groups: Vec::new(),
        }
    }

    /// Adds a test group.
    pub fn add_group(&mut self, name: String, provider_ids: Vec<String>, traffic_percentage: u8) {
        self.groups.push(AbTestGroup {
            name,
            provider_ids,
            traffic_percentage,
        });
    }

    /// Selects a group for a request (based on random distribution).
    #[must_use]
    pub fn select_group(&self) -> Option<&AbTestGroup> {
        if !self.enabled || self.groups.is_empty() {
            return None;
        }

        let mut rng = rand::rng();
        let roll: u8 = rng.random_range(0..100);

        let mut cumulative = 0u8;
        for group in &self.groups {
            cumulative = cumulative.saturating_add(group.traffic_percentage);
            if roll < cumulative {
                return Some(group);
            }
        }

        self.groups.last()
    }
}

impl Default for AbTestConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Traffic shaping configuration.
#[derive(Debug, Serialize, Deserialize)]
pub struct TrafficShaping {
    /// Enable traffic shaping.
    pub enabled: bool,
    /// Rate limits per provider (requests per second).
    pub rate_limits: HashMap<String, u64>,
    /// Current request counts.
    request_counts: HashMap<String, AtomicU64>,
    /// Last reset time.
    last_reset: SystemTime,
}

impl TrafficShaping {
    /// Creates new traffic shaping configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled: false,
            rate_limits: HashMap::new(),
            request_counts: HashMap::new(),
            last_reset: SystemTime::now(),
        }
    }

    /// Sets rate limit for a provider.
    pub fn set_rate_limit(&mut self, provider_id: String, limit: u64) {
        self.rate_limits.insert(provider_id.clone(), limit);
        self.request_counts.insert(provider_id, AtomicU64::new(0));
    }

    /// Checks if a request is allowed for a provider.
    pub fn allow_request(&mut self, provider_id: &str) -> bool {
        if !self.enabled {
            return true;
        }

        // Reset counters every second
        if let Ok(elapsed) = self.last_reset.elapsed() {
            if elapsed >= Duration::from_secs(1) {
                for counter in self.request_counts.values() {
                    counter.store(0, Ordering::Relaxed);
                }
                self.last_reset = SystemTime::now();
            }
        }

        let limit = match self.rate_limits.get(provider_id) {
            Some(&limit) => limit,
            None => return true,
        };

        let counter = match self.request_counts.get(provider_id) {
            Some(counter) => counter,
            None => return true,
        };

        let current = counter.fetch_add(1, Ordering::Relaxed);
        current < limit
    }
}

impl Default for TrafficShaping {
    fn default() -> Self {
        Self::new()
    }
}

/// Connection tracking per provider.
#[derive(Debug, Default)]
struct ConnectionTracker {
    /// Active connections per provider.
    connections: HashMap<String, AtomicU64>,
}

impl ConnectionTracker {
    /// Increments connection count for a provider.
    fn increment(&mut self, provider_id: &str) {
        self.connections
            .entry(provider_id.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Decrements connection count for a provider.
    fn decrement(&mut self, provider_id: &str) {
        if let Some(counter) = self.connections.get(provider_id) {
            counter.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Gets connection count for a provider.
    fn get_count(&self, provider_id: &str) -> u64 {
        self.connections
            .get(provider_id)
            .map_or(0, |c| c.load(Ordering::Relaxed))
    }
}

/// Router state.
struct RouterState {
    /// Current routing strategy.
    strategy: RoutingStrategy,
    /// Round-robin counter.
    round_robin_counter: AtomicU64,
    /// Session affinity manager.
    session_affinity: SessionAffinity,
    /// A/B testing configuration.
    ab_test: AbTestConfig,
    /// Traffic shaping.
    traffic_shaping: TrafficShaping,
    /// Connection tracker.
    connection_tracker: ConnectionTracker,
    /// Provider list (maintained for ordering).
    provider_list: Vec<String>,
}

/// Router for selecting CDN providers.
pub struct Router {
    /// Internal state.
    state: Arc<RwLock<RouterState>>,
}

impl Router {
    /// Creates a new router with the specified strategy.
    #[must_use]
    pub fn new(strategy: RoutingStrategy) -> Self {
        let state = RouterState {
            strategy,
            round_robin_counter: AtomicU64::new(0),
            session_affinity: SessionAffinity::new(true, Duration::from_secs(300)),
            ab_test: AbTestConfig::new(),
            traffic_shaping: TrafficShaping::new(),
            connection_tracker: ConnectionTracker::default(),
            provider_list: Vec::new(),
        };

        Self {
            state: Arc::new(RwLock::new(state)),
        }
    }

    /// Sets the routing strategy.
    pub fn set_strategy(&self, strategy: RoutingStrategy) {
        self.state.write().strategy = strategy;
    }

    /// Gets the current routing strategy.
    #[must_use]
    pub fn get_strategy(&self) -> RoutingStrategy {
        self.state.read().strategy
    }

    /// Adds a provider to the router.
    pub fn add_provider(&self, provider_id: String) {
        let mut state = self.state.write();
        if !state.provider_list.contains(&provider_id) {
            state.provider_list.push(provider_id);
        }
    }

    /// Removes a provider from the router.
    pub fn remove_provider(&self, provider_id: &str) {
        let mut state = self.state.write();
        state.provider_list.retain(|id| id != provider_id);
    }

    /// Selects a provider based on the current strategy.
    pub fn select_provider(
        &self,
        available_providers: &[String],
        context: &RequestContext,
        health_checker: &HealthChecker,
    ) -> NetResult<String> {
        if available_providers.is_empty() {
            return Err(NetError::connection("No available providers"));
        }

        let mut state = self.state.write();

        // Check session affinity first
        if let Some(session_id) = &context.session_id {
            if let Some(provider_id) = state.session_affinity.get_provider(session_id) {
                if available_providers.contains(&provider_id) {
                    return Ok(provider_id);
                }
            }
        }

        // Check A/B testing
        if state.ab_test.enabled {
            if let Some(group) = state.ab_test.select_group() {
                let group_providers: Vec<_> = group
                    .provider_ids
                    .iter()
                    .filter(|id| available_providers.contains(id))
                    .collect();

                if !group_providers.is_empty() {
                    let provider_id = self.select_by_strategy(
                        &group_providers
                            .iter()
                            .map(|s| (*s).clone())
                            .collect::<Vec<_>>(),
                        context,
                        health_checker,
                        &state,
                    )?;
                    return Ok(provider_id);
                }
            }
        }

        // Select by strategy
        let provider_id =
            self.select_by_strategy(available_providers, context, health_checker, &state)?;

        // Check traffic shaping
        if !state.traffic_shaping.allow_request(&provider_id) {
            // Find alternative provider
            for alternative in available_providers {
                if alternative != &provider_id && state.traffic_shaping.allow_request(alternative) {
                    return Ok(alternative.clone());
                }
            }
        }

        // Bind session if needed
        if let Some(session_id) = &context.session_id {
            state
                .session_affinity
                .bind_session(session_id.clone(), provider_id.clone());
        }

        Ok(provider_id)
    }

    /// Selects a provider by strategy (internal).
    #[allow(clippy::too_many_lines)]
    fn select_by_strategy(
        &self,
        available_providers: &[String],
        context: &RequestContext,
        health_checker: &HealthChecker,
        state: &RouterState,
    ) -> NetResult<String> {
        match state.strategy {
            RoutingStrategy::RoundRobin => {
                let idx = state.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
                Ok(available_providers[idx % available_providers.len()].clone())
            }

            RoutingStrategy::WeightedRoundRobin => {
                // For simplicity, use round-robin
                // In production, implement proper weighted selection
                let idx = state.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
                Ok(available_providers[idx % available_providers.len()].clone())
            }

            RoutingStrategy::LeastConnections => {
                let provider = available_providers
                    .iter()
                    .min_by_key(|id| state.connection_tracker.get_count(id))
                    .ok_or_else(|| NetError::connection("No providers available"))?;
                Ok(provider.clone())
            }

            RoutingStrategy::LeastLatency => {
                let provider = available_providers
                    .iter()
                    .min_by_key(|id| {
                        health_checker
                            .get_health(id)
                            .map_or(Duration::from_secs(999), |h| h.latency.avg)
                    })
                    .ok_or_else(|| NetError::connection("No providers available"))?;
                Ok(provider.clone())
            }

            RoutingStrategy::Geographic => {
                if let Some(region) = context.client_region {
                    if let Some(provider_id) =
                        health_checker.get_best_for_region(region, available_providers)
                    {
                        return Ok(provider_id);
                    }
                }
                // Fallback to round-robin
                let idx = state.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
                Ok(available_providers[idx % available_providers.len()].clone())
            }

            RoutingStrategy::CostBased => {
                // Select lowest cost provider
                // For now, just use round-robin as we don't have cost data easily accessible
                let idx = state.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
                Ok(available_providers[idx % available_providers.len()].clone())
            }

            RoutingStrategy::Priority => {
                // Select highest priority provider
                // For now, just use first available
                Ok(available_providers[0].clone())
            }

            RoutingStrategy::Random => {
                let mut rng = rand::rng();
                let idx = rng.random_range(0..available_providers.len());
                Ok(available_providers[idx].clone())
            }

            RoutingStrategy::HashBased => {
                // Use session ID or path for consistent hashing
                let hash_key = context.session_id.as_deref().unwrap_or(&context.path);
                let hash = self.hash_string(hash_key);
                let idx = (hash % available_providers.len() as u64) as usize;
                Ok(available_providers[idx].clone())
            }
        }
    }

    /// Simple string hash function.
    fn hash_string(&self, s: &str) -> u64 {
        let mut hash = 0u64;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        hash
    }

    /// Records a connection start.
    pub fn record_connection_start(&self, provider_id: &str) {
        self.state.write().connection_tracker.increment(provider_id);
    }

    /// Records a connection end.
    pub fn record_connection_end(&self, provider_id: &str) {
        self.state.write().connection_tracker.decrement(provider_id);
    }

    /// Gets active connection count for a provider.
    #[must_use]
    pub fn get_connection_count(&self, provider_id: &str) -> u64 {
        self.state.read().connection_tracker.get_count(provider_id)
    }

    /// Enables session affinity.
    pub fn enable_session_affinity(&self, timeout: Duration) {
        let mut state = self.state.write();
        state.session_affinity.enabled = true;
        state.session_affinity.timeout = timeout;
    }

    /// Disables session affinity.
    pub fn disable_session_affinity(&self) {
        self.state.write().session_affinity.enabled = false;
    }

    /// Configures A/B testing.
    pub fn configure_ab_testing(&self, config: AbTestConfig) {
        self.state.write().ab_test = config;
    }

    /// Configures traffic shaping.
    pub fn configure_traffic_shaping(&self, config: TrafficShaping) {
        self.state.write().traffic_shaping = config;
    }

    /// Cleans up expired sessions.
    pub fn cleanup_sessions(&self) {
        self.state.write().session_affinity.cleanup_expired();
    }

    /// Gets active session count.
    #[must_use]
    pub fn active_sessions(&self) -> usize {
        self.state.read().session_affinity.active_sessions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_strategy_names() {
        assert_eq!(RoutingStrategy::RoundRobin.name(), "Round Robin");
        assert_eq!(RoutingStrategy::LeastLatency.name(), "Least Latency");
        assert_eq!(RoutingStrategy::Geographic.name(), "Geographic");
    }

    #[test]
    fn test_session_affinity() {
        let mut affinity = SessionAffinity::new(true, Duration::from_secs(300));

        affinity.bind_session("session-1".to_string(), "provider-1".to_string());
        assert_eq!(
            affinity.get_provider("session-1"),
            Some("provider-1".to_string())
        );
        assert_eq!(affinity.active_sessions(), 1);
    }

    #[test]
    fn test_session_affinity_disabled() {
        let mut affinity = SessionAffinity::new(false, Duration::from_secs(300));

        affinity.bind_session("session-1".to_string(), "provider-1".to_string());
        assert_eq!(affinity.get_provider("session-1"), None);
    }

    #[test]
    fn test_ab_test_config() {
        let mut config = AbTestConfig::new();
        config.add_group("group-a".to_string(), vec!["provider-1".to_string()], 50);
        config.add_group("group-b".to_string(), vec!["provider-2".to_string()], 50);

        assert_eq!(config.groups.len(), 2);
    }

    #[test]
    fn test_traffic_shaping() {
        let mut shaping = TrafficShaping::new();
        shaping.enabled = true;
        shaping.set_rate_limit("provider-1".to_string(), 10);

        // Should allow first 10 requests
        for _ in 0..10 {
            assert!(shaping.allow_request("provider-1"));
        }

        // Should deny 11th request
        assert!(!shaping.allow_request("provider-1"));
    }

    #[test]
    fn test_router_creation() {
        let router = Router::new(RoutingStrategy::RoundRobin);
        assert_eq!(router.get_strategy(), RoutingStrategy::RoundRobin);
    }

    #[test]
    fn test_router_add_provider() {
        let router = Router::new(RoutingStrategy::RoundRobin);
        router.add_provider("provider-1".to_string());
        router.add_provider("provider-2".to_string());

        let state = router.state.read();
        assert_eq!(state.provider_list.len(), 2);
    }

    #[test]
    fn test_router_remove_provider() {
        let router = Router::new(RoutingStrategy::RoundRobin);
        router.add_provider("provider-1".to_string());
        router.add_provider("provider-2".to_string());
        router.remove_provider("provider-1");

        let state = router.state.read();
        assert_eq!(state.provider_list.len(), 1);
        assert_eq!(state.provider_list[0], "provider-2");
    }

    #[test]
    fn test_connection_tracking() {
        let router = Router::new(RoutingStrategy::RoundRobin);
        router.record_connection_start("provider-1");
        router.record_connection_start("provider-1");

        assert_eq!(router.get_connection_count("provider-1"), 2);

        router.record_connection_end("provider-1");
        assert_eq!(router.get_connection_count("provider-1"), 1);
    }

    #[test]
    fn test_session_affinity_integration() {
        let router = Router::new(RoutingStrategy::RoundRobin);
        router.enable_session_affinity(Duration::from_secs(300));

        assert!(router.active_sessions() == 0);
    }

    #[tokio::test]
    async fn test_round_robin_selection() {
        let router = Router::new(RoutingStrategy::RoundRobin);
        let health_checker = HealthChecker::new(Duration::from_secs(5), Duration::from_secs(3));

        let providers = vec!["provider-1".to_string(), "provider-2".to_string()];
        let context = RequestContext::new("/test");

        let first = router
            .select_provider(&providers, &context, &health_checker)
            .expect("Should select provider");
        let second = router
            .select_provider(&providers, &context, &health_checker)
            .expect("Should select provider");

        // With 2 providers, selections should alternate
        assert_ne!(first, second);
    }
}
