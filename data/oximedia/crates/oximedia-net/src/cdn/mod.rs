//! Multi-CDN failover and load balancing for OxiMedia streaming.
//!
//! This module provides comprehensive CDN management with support for multiple providers,
//! automatic failover, health monitoring, and intelligent routing strategies.
//!
//! # Features
//!
//! - **Multi-CDN Support**: Cloudflare, Fastly, Akamai, CloudFront, and custom providers
//! - **Real-time Health Monitoring**: Continuous health checks with sub-second failover
//! - **Intelligent Routing**: Multiple strategies including geographic, latency-based, and cost-optimized
//! - **Automatic Failover**: Circuit breaker pattern with exponential backoff
//! - **Performance Metrics**: Comprehensive monitoring with Prometheus export
//!
//! # Example
//!
//! ```ignore
//! use oximedia_net::cdn::{CdnManager, CdnProvider, CdnConfig};
//!
//! async fn setup_cdn() -> Result<CdnManager, NetError> {
//!     let config = CdnConfig::default();
//!     let mut manager = CdnManager::new(config);
//!
//!     manager.add_provider(CdnProvider::cloudflare("cdn.example.com", 100));
//!     manager.add_provider(CdnProvider::fastly("fastly.example.com", 90));
//!
//!     manager.start_health_monitoring().await?;
//!     Ok(manager)
//! }
//! ```

use crate::error::{NetError, NetResult};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

pub mod failover;
pub mod health;
pub mod metrics;
pub mod routing;

pub use failover::{CircuitBreaker, CircuitState, FailoverManager};
pub use health::{HealthChecker, HealthStatus, ProviderHealth};
pub use metrics::{CdnMetrics, MetricsCollector, PerformanceMetrics};
pub use routing::{Router, RoutingStrategy, SessionAffinity};

/// CDN provider types supported by the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CdnProviderType {
    /// Cloudflare CDN.
    Cloudflare,
    /// Fastly CDN.
    Fastly,
    /// Akamai CDN.
    Akamai,
    /// AWS CloudFront.
    CloudFront,
    /// Generic CDN provider.
    Generic,
    /// Custom CDN implementation.
    Custom,
}

impl CdnProviderType {
    /// Returns the display name of the provider.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Cloudflare => "Cloudflare",
            Self::Fastly => "Fastly",
            Self::Akamai => "Akamai",
            Self::CloudFront => "CloudFront",
            Self::Generic => "Generic",
            Self::Custom => "Custom",
        }
    }

    /// Returns the default port for the provider.
    #[must_use]
    pub const fn default_port(&self) -> u16 {
        match self {
            Self::Cloudflare | Self::Fastly | Self::Akamai | Self::CloudFront | Self::Generic => {
                443
            }
            Self::Custom => 8080,
        }
    }
}

/// Geographic region for CDN selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Region {
    /// North America.
    NorthAmerica,
    /// South America.
    SouthAmerica,
    /// Europe.
    Europe,
    /// Asia Pacific.
    AsiaPacific,
    /// Middle East.
    MiddleEast,
    /// Africa.
    Africa,
    /// Oceania.
    Oceania,
}

impl Region {
    /// Returns all available regions.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::NorthAmerica,
            Self::SouthAmerica,
            Self::Europe,
            Self::AsiaPacific,
            Self::MiddleEast,
            Self::Africa,
            Self::Oceania,
        ]
    }

    /// Returns the region name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::NorthAmerica => "North America",
            Self::SouthAmerica => "South America",
            Self::Europe => "Europe",
            Self::AsiaPacific => "Asia Pacific",
            Self::MiddleEast => "Middle East",
            Self::Africa => "Africa",
            Self::Oceania => "Oceania",
        }
    }
}

/// CDN provider configuration and state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnProvider {
    /// Unique identifier for this provider instance.
    pub id: String,
    /// Provider type.
    pub provider_type: CdnProviderType,
    /// Base URL for the CDN endpoint.
    pub base_url: String,
    /// Provider priority (0-100, higher is better).
    pub priority: u8,
    /// Weight for weighted routing (0-100).
    pub weight: u8,
    /// Geographic region.
    pub region: Region,
    /// Whether the provider is enabled.
    pub enabled: bool,
    /// Cost per GB (in cents).
    pub cost_per_gb: f64,
    /// Maximum bandwidth (Mbps).
    pub max_bandwidth: u64,
    /// Provider-specific configuration.
    pub config: HashMap<String, String>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
}

impl CdnProvider {
    /// Creates a new CDN provider.
    #[must_use]
    pub fn new(provider_type: CdnProviderType, base_url: impl Into<String>, priority: u8) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            provider_type,
            base_url: base_url.into(),
            priority,
            weight: 50,
            region: Region::NorthAmerica,
            enabled: true,
            cost_per_gb: 0.05,
            max_bandwidth: 10_000,
            config: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// Creates a Cloudflare CDN provider.
    #[must_use]
    pub fn cloudflare(base_url: impl Into<String>, priority: u8) -> Self {
        Self::new(CdnProviderType::Cloudflare, base_url, priority)
    }

    /// Creates a Fastly CDN provider.
    #[must_use]
    pub fn fastly(base_url: impl Into<String>, priority: u8) -> Self {
        Self::new(CdnProviderType::Fastly, base_url, priority)
    }

    /// Creates an Akamai CDN provider.
    #[must_use]
    pub fn akamai(base_url: impl Into<String>, priority: u8) -> Self {
        Self::new(CdnProviderType::Akamai, base_url, priority)
    }

    /// Creates a CloudFront CDN provider.
    #[must_use]
    pub fn cloudfront(base_url: impl Into<String>, priority: u8) -> Self {
        Self::new(CdnProviderType::CloudFront, base_url, priority)
    }

    /// Creates a custom CDN provider.
    #[must_use]
    pub fn custom(base_url: impl Into<String>, priority: u8) -> Self {
        Self::new(CdnProviderType::Custom, base_url, priority)
    }

    /// Sets the provider region.
    #[must_use]
    pub fn with_region(mut self, region: Region) -> Self {
        self.region = region;
        self
    }

    /// Sets the provider weight.
    #[must_use]
    pub fn with_weight(mut self, weight: u8) -> Self {
        self.weight = weight;
        self
    }

    /// Sets the cost per GB.
    #[must_use]
    pub fn with_cost(mut self, cost_per_gb: f64) -> Self {
        self.cost_per_gb = cost_per_gb;
        self
    }

    /// Sets the maximum bandwidth.
    #[must_use]
    pub fn with_bandwidth(mut self, max_bandwidth: u64) -> Self {
        self.max_bandwidth = max_bandwidth;
        self
    }

    /// Sets a configuration value.
    pub fn set_config(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.config.insert(key.into(), value.into());
    }

    /// Gets a configuration value.
    #[must_use]
    pub fn get_config(&self, key: &str) -> Option<&str> {
        self.config.get(key).map(String::as_str)
    }

    /// Builds the full URL for a path.
    #[must_use]
    pub fn build_url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        let path = path.trim_start_matches('/');
        format!("{base}/{path}")
    }
}

/// CDN manager configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnConfig {
    /// Health check interval.
    pub health_check_interval: Duration,
    /// Health check timeout.
    pub health_check_timeout: Duration,
    /// Failover threshold (number of consecutive failures).
    pub failover_threshold: u32,
    /// Recovery check interval.
    pub recovery_check_interval: Duration,
    /// Default routing strategy.
    pub default_routing_strategy: RoutingStrategy,
    /// Enable session affinity.
    pub enable_session_affinity: bool,
    /// Session timeout.
    pub session_timeout: Duration,
    /// Metrics collection interval.
    pub metrics_interval: Duration,
    /// Enable automatic failover.
    pub enable_auto_failover: bool,
    /// Circuit breaker open timeout.
    pub circuit_breaker_timeout: Duration,
}

impl Default for CdnConfig {
    fn default() -> Self {
        Self {
            health_check_interval: Duration::from_secs(5),
            health_check_timeout: Duration::from_secs(3),
            failover_threshold: 3,
            recovery_check_interval: Duration::from_secs(30),
            default_routing_strategy: RoutingStrategy::LeastLatency,
            enable_session_affinity: true,
            session_timeout: Duration::from_secs(300),
            metrics_interval: Duration::from_secs(10),
            enable_auto_failover: true,
            circuit_breaker_timeout: Duration::from_secs(60),
        }
    }
}

/// Request context for CDN routing decisions.
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Request path.
    pub path: String,
    /// Client region.
    pub client_region: Option<Region>,
    /// Session ID for sticky sessions.
    pub session_id: Option<String>,
    /// Request priority.
    pub priority: u8,
    /// Expected content size (bytes).
    pub expected_size: Option<u64>,
    /// Custom metadata.
    pub metadata: HashMap<String, String>,
}

impl RequestContext {
    /// Creates a new request context.
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            client_region: None,
            session_id: None,
            priority: 50,
            expected_size: None,
            metadata: HashMap::new(),
        }
    }

    /// Sets the client region.
    #[must_use]
    pub fn with_region(mut self, region: Region) -> Self {
        self.client_region = Some(region);
        self
    }

    /// Sets the session ID.
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Sets the priority.
    #[must_use]
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

/// CDN manager state.
struct CdnManagerState {
    /// Registered CDN providers.
    providers: HashMap<String, CdnProvider>,
    /// Health checker.
    health_checker: HealthChecker,
    /// Router.
    router: Router,
    /// Failover manager.
    failover_manager: FailoverManager,
    /// Metrics collector.
    metrics: MetricsCollector,
}

/// Multi-CDN manager with automatic failover and load balancing.
pub struct CdnManager {
    /// Configuration.
    config: CdnConfig,
    /// Internal state.
    state: Arc<RwLock<CdnManagerState>>,
    /// Shutdown channel.
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl CdnManager {
    /// Creates a new CDN manager.
    #[must_use]
    pub fn new(config: CdnConfig) -> Self {
        let state = CdnManagerState {
            providers: HashMap::new(),
            health_checker: HealthChecker::new(
                config.health_check_interval,
                config.health_check_timeout,
            ),
            router: Router::new(config.default_routing_strategy),
            failover_manager: FailoverManager::new(
                config.failover_threshold,
                config.circuit_breaker_timeout,
            ),
            metrics: MetricsCollector::new(config.metrics_interval),
        };

        Self {
            config,
            state: Arc::new(RwLock::new(state)),
            shutdown_tx: None,
        }
    }

    /// Adds a CDN provider.
    pub fn add_provider(&self, provider: CdnProvider) {
        let mut state = self.state.write();
        let id = provider.id.clone();
        state.providers.insert(id.clone(), provider.clone());
        state.health_checker.add_provider(id.clone(), provider);
        state.router.add_provider(id);
    }

    /// Removes a CDN provider.
    pub fn remove_provider(&self, provider_id: &str) -> bool {
        let mut state = self.state.write();
        if state.providers.remove(provider_id).is_some() {
            state.health_checker.remove_provider(provider_id);
            state.router.remove_provider(provider_id);
            true
        } else {
            false
        }
    }

    /// Gets a provider by ID.
    #[must_use]
    pub fn get_provider(&self, provider_id: &str) -> Option<CdnProvider> {
        self.state.read().providers.get(provider_id).cloned()
    }

    /// Lists all providers.
    #[must_use]
    pub fn list_providers(&self) -> Vec<CdnProvider> {
        self.state.read().providers.values().cloned().collect()
    }

    /// Gets the health status of a provider.
    #[must_use]
    pub fn get_health(&self, provider_id: &str) -> Option<ProviderHealth> {
        self.state.read().health_checker.get_health(provider_id)
    }

    /// Starts background health monitoring and metrics collection.
    pub async fn start(&mut self) -> NetResult<()> {
        let (tx, mut rx) = mpsc::channel::<()>(1);
        self.shutdown_tx = Some(tx);

        let state = Arc::clone(&self.state);
        let config = self.config.clone();

        // Start health monitoring task
        let health_state = Arc::clone(&state);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.health_check_interval);
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let state = health_state.read();
                        let providers: Vec<_> = state.providers.keys().cloned().collect();
                        drop(state);

                        for provider_id in providers {
                            let state = health_state.read();
                            if let Some(health) = state.health_checker.get_health(&provider_id) {
                                // Update failover manager with health status
                                if health.status != HealthStatus::Healthy {
                                    drop(state);
                                    let state = health_state.write();
                                    state.failover_manager.record_failure(&provider_id);
                                }
                            }
                        }
                    }
                    _ = rx.recv() => {
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Stops background tasks.
    pub async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _result = tx.send(()).await;
        }
    }

    /// Routes a request to an appropriate CDN provider.
    pub async fn route_request(&self, context: &RequestContext) -> NetResult<String> {
        let state = self.state.read();

        // Get available healthy providers
        let available_providers: Vec<String> = state
            .providers
            .keys()
            .filter(|id| {
                state
                    .health_checker
                    .get_health(id)
                    .map_or(false, |h| h.status == HealthStatus::Healthy)
                    && !state.failover_manager.is_open(id)
            })
            .cloned()
            .collect();

        if available_providers.is_empty() {
            return Err(NetError::connection("No healthy CDN providers available"));
        }

        // Select provider based on routing strategy
        let provider_id =
            state
                .router
                .select_provider(&available_providers, context, &state.health_checker)?;

        // Get the provider and build URL
        let provider = state
            .providers
            .get(&provider_id)
            .ok_or_else(|| NetError::connection("Provider not found"))?;

        let url = provider.build_url(&context.path);

        // Record metrics
        drop(state);
        let state = self.state.write();
        state.metrics.record_request(&provider_id);

        Ok(url)
    }

    /// Records the result of a request.
    pub fn record_result(
        &self,
        provider_id: &str,
        success: bool,
        latency: Duration,
        bytes_transferred: u64,
    ) {
        let state = self.state.write();

        if success {
            state.failover_manager.record_success(provider_id);
            state
                .metrics
                .record_success(provider_id, latency, bytes_transferred);
        } else {
            state.failover_manager.record_failure(provider_id);
            state.metrics.record_failure(provider_id);
        }

        // Update health checker with latency
        state.health_checker.record_latency(provider_id, latency);
    }

    /// Gets performance metrics for all providers.
    #[must_use]
    pub fn get_metrics(&self) -> HashMap<String, PerformanceMetrics> {
        self.state.read().metrics.get_all_metrics()
    }

    /// Gets performance metrics for a specific provider.
    #[must_use]
    pub fn get_provider_metrics(&self, provider_id: &str) -> Option<PerformanceMetrics> {
        self.state.read().metrics.get_metrics(provider_id)
    }

    /// Sets the routing strategy.
    pub fn set_routing_strategy(&self, strategy: RoutingStrategy) {
        self.state.write().router.set_strategy(strategy);
    }

    /// Forces a provider into maintenance mode.
    pub fn set_maintenance(&self, provider_id: &str, maintenance: bool) {
        let mut state = self.state.write();
        if let Some(provider) = state.providers.get_mut(provider_id) {
            provider.enabled = !maintenance;
        }
    }

    /// Resets circuit breakers for all providers.
    pub fn reset_circuit_breakers(&self) {
        self.state.write().failover_manager.reset_all();
    }
}

impl Drop for CdnManager {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _result = tx.try_send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdn_provider_creation() {
        let provider = CdnProvider::cloudflare("https://cdn.example.com", 100);
        assert_eq!(provider.provider_type, CdnProviderType::Cloudflare);
        assert_eq!(provider.base_url, "https://cdn.example.com");
        assert_eq!(provider.priority, 100);
    }

    #[test]
    fn test_cdn_provider_url_building() {
        let provider = CdnProvider::fastly("https://fastly.example.com", 90);
        let url = provider.build_url("/video/stream.m3u8");
        assert_eq!(url, "https://fastly.example.com/video/stream.m3u8");
    }

    #[test]
    fn test_cdn_provider_with_region() {
        let provider = CdnProvider::akamai("https://akamai.example.com", 80)
            .with_region(Region::Europe)
            .with_weight(75);
        assert_eq!(provider.region, Region::Europe);
        assert_eq!(provider.weight, 75);
    }

    #[test]
    fn test_request_context() {
        let context = RequestContext::new("/video/test.mp4")
            .with_region(Region::AsiaPacific)
            .with_session("session-123")
            .with_priority(90);

        assert_eq!(context.path, "/video/test.mp4");
        assert_eq!(context.client_region, Some(Region::AsiaPacific));
        assert_eq!(context.session_id.as_deref(), Some("session-123"));
        assert_eq!(context.priority, 90);
    }

    #[test]
    fn test_cdn_manager_creation() {
        let config = CdnConfig::default();
        let manager = CdnManager::new(config);
        assert_eq!(manager.list_providers().len(), 0);
    }

    #[test]
    fn test_cdn_manager_add_provider() {
        let config = CdnConfig::default();
        let manager = CdnManager::new(config);

        let provider = CdnProvider::cloudflare("https://cdn1.example.com", 100);
        let provider_id = provider.id.clone();

        manager.add_provider(provider);
        assert_eq!(manager.list_providers().len(), 1);
        assert!(manager.get_provider(&provider_id).is_some());
    }

    #[test]
    fn test_cdn_manager_remove_provider() {
        let config = CdnConfig::default();
        let manager = CdnManager::new(config);

        let provider = CdnProvider::fastly("https://cdn2.example.com", 90);
        let provider_id = provider.id.clone();

        manager.add_provider(provider);
        assert!(manager.remove_provider(&provider_id));
        assert_eq!(manager.list_providers().len(), 0);
    }

    #[test]
    fn test_region_names() {
        assert_eq!(Region::NorthAmerica.name(), "North America");
        assert_eq!(Region::Europe.name(), "Europe");
        assert_eq!(Region::AsiaPacific.name(), "Asia Pacific");
    }

    #[test]
    fn test_provider_type_names() {
        assert_eq!(CdnProviderType::Cloudflare.name(), "Cloudflare");
        assert_eq!(CdnProviderType::Fastly.name(), "Fastly");
        assert_eq!(CdnProviderType::Akamai.name(), "Akamai");
        assert_eq!(CdnProviderType::CloudFront.name(), "CloudFront");
    }
}
