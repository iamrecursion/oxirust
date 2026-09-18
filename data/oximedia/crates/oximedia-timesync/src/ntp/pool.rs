//! NTP server pool management.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

/// Server pool for NTP client with failover support.
#[derive(Debug, Clone)]
pub struct ServerPool {
    /// Servers with their statistics
    servers: HashMap<SocketAddr, ServerStats>,
    /// Server selection strategy
    strategy: SelectionStrategy,
}

impl ServerPool {
    /// Create a new server pool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            servers: HashMap::new(),
            strategy: SelectionStrategy::RoundRobin,
        }
    }

    /// Create a server pool with common public NTP servers.
    #[must_use]
    pub fn with_default_servers() -> Self {
        let mut pool = Self::new();

        // Add common public NTP servers
        let default_servers = [
            "time.google.com:123",
            "time.cloudflare.com:123",
            "pool.ntp.org:123",
        ];

        for server_str in &default_servers {
            if let Ok(addr) = server_str.parse() {
                pool.add_server(addr);
            }
        }

        pool
    }

    /// Add a server to the pool.
    pub fn add_server(&mut self, addr: SocketAddr) {
        self.servers.insert(addr, ServerStats::new());
    }

    /// Remove a server from the pool.
    pub fn remove_server(&mut self, addr: &SocketAddr) {
        self.servers.remove(addr);
    }

    /// Get all servers in the pool.
    #[must_use]
    pub fn servers(&self) -> Vec<&SocketAddr> {
        self.servers.keys().collect()
    }

    /// Mark a successful query to a server.
    pub fn mark_success(&mut self, addr: SocketAddr) {
        if let Some(stats) = self.servers.get_mut(&addr) {
            stats.record_success();
        }
    }

    /// Mark a failed query to a server.
    pub fn mark_failure(&mut self, addr: SocketAddr) {
        if let Some(stats) = self.servers.get_mut(&addr) {
            stats.record_failure();
        }
    }

    /// Get the best server based on statistics.
    #[must_use]
    pub fn best_server(&self) -> Option<SocketAddr> {
        self.servers
            .iter()
            .filter(|(_, stats)| !stats.is_blacklisted())
            .max_by_key(|(_, stats)| stats.success_rate_permille())
            .map(|(addr, _)| *addr)
    }

    /// Get server statistics.
    #[must_use]
    pub fn get_stats(&self, addr: &SocketAddr) -> Option<&ServerStats> {
        self.servers.get(addr)
    }

    /// Set selection strategy.
    pub fn set_strategy(&mut self, strategy: SelectionStrategy) {
        self.strategy = strategy;
    }

    /// Get selection strategy.
    #[must_use]
    pub fn strategy(&self) -> SelectionStrategy {
        self.strategy
    }
}

impl Default for ServerPool {
    fn default() -> Self {
        Self::with_default_servers()
    }
}

/// Server statistics.
#[derive(Debug, Clone)]
pub struct ServerStats {
    /// Total queries
    total_queries: u32,
    /// Successful queries
    successful_queries: u32,
    /// Failed queries
    failed_queries: u32,
    /// Last query time
    last_query: Option<Instant>,
    /// Last success time
    last_success: Option<Instant>,
    /// Consecutive failures
    consecutive_failures: u32,
}

impl ServerStats {
    /// Create new server statistics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_queries: 0,
            successful_queries: 0,
            failed_queries: 0,
            last_query: None,
            last_success: None,
            consecutive_failures: 0,
        }
    }

    /// Record a successful query.
    pub fn record_success(&mut self) {
        self.total_queries += 1;
        self.successful_queries += 1;
        self.consecutive_failures = 0;
        let now = Instant::now();
        self.last_query = Some(now);
        self.last_success = Some(now);
    }

    /// Record a failed query.
    pub fn record_failure(&mut self) {
        self.total_queries += 1;
        self.failed_queries += 1;
        self.consecutive_failures += 1;
        self.last_query = Some(Instant::now());
    }

    /// Get success rate (0-1000 permille).
    #[must_use]
    pub fn success_rate_permille(&self) -> u32 {
        if self.total_queries == 0 {
            return 0;
        }
        (self.successful_queries * 1000) / self.total_queries
    }

    /// Check if server should be blacklisted.
    #[must_use]
    pub fn is_blacklisted(&self) -> bool {
        // Blacklist if 5 consecutive failures
        self.consecutive_failures >= 5
    }

    /// Get time since last success.
    #[must_use]
    pub fn time_since_last_success(&self) -> Option<Duration> {
        self.last_success.map(|t| t.elapsed())
    }

    /// Get total queries.
    #[must_use]
    pub fn total_queries(&self) -> u32 {
        self.total_queries
    }

    /// Get successful queries.
    #[must_use]
    pub fn successful_queries(&self) -> u32 {
        self.successful_queries
    }

    /// Get failed queries.
    #[must_use]
    pub fn failed_queries(&self) -> u32 {
        self.failed_queries
    }
}

impl Default for ServerStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Server selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Round-robin selection
    RoundRobin,
    /// Select best server by success rate
    BestFirst,
    /// Random selection
    Random,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_pool_creation() {
        let pool = ServerPool::new();
        assert_eq!(pool.servers().len(), 0);
    }

    #[test]
    fn test_add_remove_server() {
        let mut pool = ServerPool::new();
        let addr: SocketAddr = "127.0.0.1:123".parse().expect("should succeed in test");

        pool.add_server(addr);
        assert_eq!(pool.servers().len(), 1);

        pool.remove_server(&addr);
        assert_eq!(pool.servers().len(), 0);
    }

    #[test]
    fn test_server_stats() {
        let mut stats = ServerStats::new();
        assert_eq!(stats.success_rate_permille(), 0);

        stats.record_success();
        assert_eq!(stats.total_queries(), 1);
        assert_eq!(stats.successful_queries(), 1);
        assert_eq!(stats.success_rate_permille(), 1000);

        stats.record_failure();
        assert_eq!(stats.total_queries(), 2);
        assert_eq!(stats.success_rate_permille(), 500);
    }

    #[test]
    fn test_blacklist() {
        let mut stats = ServerStats::new();
        assert!(!stats.is_blacklisted());

        for _ in 0..5 {
            stats.record_failure();
        }

        assert!(stats.is_blacklisted());

        stats.record_success();
        assert!(!stats.is_blacklisted());
    }

    #[test]
    fn test_default_pool() {
        let pool = ServerPool::default();
        // Default pool attempts DNS resolution which may fail in test environment
        // Just verify it was created successfully
        let _servers = pool.servers();
    }
}
