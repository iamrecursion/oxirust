//! Automatic failover routing for resilient media transport.
//!
//! Provides health-aware route management with automatic failover to standby
//! paths when the primary route becomes unavailable.

#![allow(dead_code)]

/// Health state of a route in the failover system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouteHealth {
    /// The route is the currently active primary path.
    #[default]
    Primary,
    /// The route is on standby, ready to take over.
    Standby,
    /// The route has failed and is not usable.
    Failed,
    /// The route is recovering and being tested for usability.
    Recovering,
}

impl RouteHealth {
    /// Returns `true` if this route can carry traffic.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Primary | Self::Standby)
    }
}

/// A single route entry in the failover manager.
#[derive(Debug, Clone)]
pub struct FailoverRoute {
    /// Unique route identifier.
    pub id: u32,
    /// Path identifier or address for this route.
    pub path: String,
    /// Current health state.
    pub health: RouteHealth,
    /// Priority — lower value means higher priority (0 = highest).
    pub priority: u8,
    /// Number of times this route has failed.
    pub fail_count: u32,
}

impl FailoverRoute {
    /// Moves the route to `Primary` state, indicating it is actively carrying traffic.
    pub fn activate(&mut self) {
        self.health = RouteHealth::Primary;
    }

    /// Moves the route to `Standby` state (ready but not active).
    pub fn deactivate(&mut self) {
        self.health = RouteHealth::Standby;
    }

    /// Marks the route as `Failed` and increments the fail counter.
    pub fn mark_failed(&mut self) {
        self.health = RouteHealth::Failed;
        self.fail_count += 1;
    }

    /// Returns `true` if the route is currently the active primary.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.health == RouteHealth::Primary
    }
}

/// Configuration parameters for the failover system.
#[derive(Debug, Clone)]
pub struct FailoverConfig {
    /// Time in milliseconds before a route is declared failed after a loss of signal.
    pub detection_ms: u32,
    /// Time in milliseconds before a failed route is allowed to recover.
    pub recovery_ms: u32,
    /// Maximum number of consecutive failures before a route is permanently excluded.
    pub max_fail_count: u32,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            detection_ms: 200,
            recovery_ms: 5000,
            max_fail_count: 5,
        }
    }
}

/// Manages a pool of routes with automatic priority-based failover.
#[derive(Debug)]
pub struct FailoverManager {
    routes: Vec<FailoverRoute>,
    config: FailoverConfig,
    /// ID of the currently active route, if any.
    pub active_route: Option<u32>,
}

impl FailoverManager {
    /// Creates a new `FailoverManager` with the given configuration.
    #[must_use]
    pub fn new(config: FailoverConfig) -> Self {
        Self {
            routes: Vec::new(),
            config,
            active_route: None,
        }
    }

    /// Adds a new route with the given path and priority. New routes start in `Standby`.
    pub fn add_route(&mut self, path: impl Into<String>, priority: u8) {
        let id = self.routes.len() as u32;
        self.routes.push(FailoverRoute {
            id,
            path: path.into(),
            health: RouteHealth::Standby,
            priority,
            fail_count: 0,
        });
    }

    /// Triggers a failover: the current active route (if any) is marked failed,
    /// and the highest-priority usable standby route is activated.
    ///
    /// Returns the ID of the newly activated route, or `None` if no standby is available.
    pub fn failover(&mut self) -> Option<u32> {
        // Mark the current active route as failed
        if let Some(active_id) = self.active_route {
            if let Some(route) = self.routes.iter_mut().find(|r| r.id == active_id) {
                route.mark_failed();
            }
        }
        self.active_route = None;

        // Pick the best usable standby route by priority (ascending = higher priority)
        let best_id = self
            .routes
            .iter()
            .filter(|r| r.health.is_usable() && !r.is_active())
            .filter(|r| r.fail_count < self.config.max_fail_count)
            .min_by_key(|r| r.priority)
            .map(|r| r.id);

        if let Some(id) = best_id {
            if let Some(route) = self.routes.iter_mut().find(|r| r.id == id) {
                route.activate();
            }
            self.active_route = Some(id);
        }

        self.active_route
    }

    /// Returns the path string of the currently active route, if any.
    #[must_use]
    pub fn active_path(&self) -> Option<&str> {
        let active_id = self.active_route?;
        self.routes
            .iter()
            .find(|r| r.id == active_id)
            .map(|r| r.path.as_str())
    }

    /// Returns the total number of routes managed.
    #[must_use]
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_health_primary_usable() {
        assert!(RouteHealth::Primary.is_usable());
    }

    #[test]
    fn test_route_health_standby_usable() {
        assert!(RouteHealth::Standby.is_usable());
    }

    #[test]
    fn test_route_health_failed_not_usable() {
        assert!(!RouteHealth::Failed.is_usable());
    }

    #[test]
    fn test_route_health_recovering_not_usable() {
        assert!(!RouteHealth::Recovering.is_usable());
    }

    #[test]
    fn test_failover_route_activate() {
        let mut r = FailoverRoute {
            id: 0,
            path: "10.0.0.1:1234".to_string(),
            health: RouteHealth::Standby,
            priority: 0,
            fail_count: 0,
        };
        r.activate();
        assert!(r.is_active());
    }

    #[test]
    fn test_failover_route_deactivate() {
        let mut r = FailoverRoute {
            id: 0,
            path: "10.0.0.1:1234".to_string(),
            health: RouteHealth::Primary,
            priority: 0,
            fail_count: 0,
        };
        r.deactivate();
        assert!(!r.is_active());
        assert_eq!(r.health, RouteHealth::Standby);
    }

    #[test]
    fn test_failover_route_mark_failed_increments() {
        let mut r = FailoverRoute {
            id: 0,
            path: "10.0.0.1:1234".to_string(),
            health: RouteHealth::Primary,
            priority: 0,
            fail_count: 2,
        };
        r.mark_failed();
        assert_eq!(r.fail_count, 3);
        assert_eq!(r.health, RouteHealth::Failed);
    }

    #[test]
    fn test_failover_config_default() {
        let cfg = FailoverConfig::default();
        assert_eq!(cfg.detection_ms, 200);
        assert_eq!(cfg.recovery_ms, 5000);
        assert_eq!(cfg.max_fail_count, 5);
    }

    #[test]
    fn test_manager_add_route_increments_count() {
        let mut mgr = FailoverManager::new(FailoverConfig::default());
        mgr.add_route("path-a", 0);
        mgr.add_route("path-b", 1);
        assert_eq!(mgr.route_count(), 2);
    }

    #[test]
    fn test_manager_failover_picks_lowest_priority_value() {
        let mut mgr = FailoverManager::new(FailoverConfig::default());
        mgr.add_route("primary", 5); // id 0
        mgr.add_route("standby-a", 2); // id 1 — higher priority (lower value)
        mgr.add_route("standby-b", 8); // id 2

        let new_id = mgr.failover();
        assert_eq!(new_id, Some(1)); // standby-a has lowest priority value
    }

    #[test]
    fn test_manager_active_path() {
        let mut mgr = FailoverManager::new(FailoverConfig::default());
        mgr.add_route("primary-path", 0);
        mgr.failover();
        assert_eq!(mgr.active_path(), Some("primary-path"));
    }

    #[test]
    fn test_manager_failover_no_routes_returns_none() {
        let mut mgr = FailoverManager::new(FailoverConfig::default());
        assert!(mgr.failover().is_none());
    }

    #[test]
    fn test_manager_sequential_failover() {
        let mut mgr = FailoverManager::new(FailoverConfig::default());
        mgr.add_route("primary", 0); // id 0
        mgr.add_route("standby", 1); // id 1

        // First failover: activates route 0 (only one with no active)
        mgr.failover();
        assert_eq!(mgr.active_path(), Some("primary"));

        // Second failover: primary fails, standby takes over
        mgr.failover();
        assert_eq!(mgr.active_path(), Some("standby"));
    }

    #[test]
    fn test_manager_fails_over_when_max_fail_count_exceeded() {
        let mut mgr = FailoverManager::new(FailoverConfig {
            max_fail_count: 2,
            ..FailoverConfig::default()
        });
        mgr.add_route("fragile", 0); // id 0
        mgr.add_route("robust", 1); // id 1

        // Simulate fragile route exceeding fail count
        if let Some(r) = mgr.routes.iter_mut().find(|r| r.id == 0) {
            r.fail_count = 2; // at max
        }

        // failover should skip route 0 and pick route 1
        let id = mgr.failover();
        assert_eq!(id, Some(1));
    }
}
