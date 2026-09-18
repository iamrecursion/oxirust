#![allow(dead_code)]
//! Audio and video signal routing between timeline tracks.
//!
//! Provides a routing matrix that connects source tracks to destination tracks,
//! supporting side-chains, send/return buses, and sub-mix groups within
//! the timeline editing environment.

use std::collections::HashMap;

/// Unique identifier for a routing node in the signal graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RouteNodeId(u64);

impl RouteNodeId {
    /// Create a new route node identifier.
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Return the raw numeric value.
    #[must_use]
    pub fn value(self) -> u64 {
        self.0
    }
}

/// Kind of signal carried by a route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalKind {
    /// Mono audio signal.
    AudioMono,
    /// Stereo audio signal.
    AudioStereo,
    /// Surround 5.1 audio signal.
    AudioSurround51,
    /// Surround 7.1 audio signal.
    AudioSurround71,
    /// Video signal.
    Video,
    /// Data / metadata signal.
    Data,
}

/// A single route connecting a source node to a destination node.
#[derive(Debug, Clone)]
pub struct Route {
    /// Unique identifier for this route.
    pub id: u64,
    /// Source node.
    pub source: RouteNodeId,
    /// Destination node.
    pub destination: RouteNodeId,
    /// Type of signal.
    pub signal_kind: SignalKind,
    /// Gain applied to the signal in the range 0.0..=1.0.
    pub gain: f64,
    /// Whether this route is currently enabled.
    pub enabled: bool,
    /// Optional label for display purposes.
    pub label: Option<String>,
}

impl Route {
    /// Create a new route with default gain of 1.0.
    #[must_use]
    pub fn new(
        id: u64,
        source: RouteNodeId,
        destination: RouteNodeId,
        signal_kind: SignalKind,
    ) -> Self {
        Self {
            id,
            source,
            destination,
            signal_kind,
            gain: 1.0,
            enabled: true,
            label: None,
        }
    }

    /// Set gain and return self for chaining.
    #[must_use]
    pub fn with_gain(mut self, gain: f64) -> Self {
        self.gain = gain.clamp(0.0, 1.0);
        self
    }

    /// Set the label and return self for chaining.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A bus that aggregates signals from multiple sources.
#[derive(Debug, Clone)]
pub struct RouteBus {
    /// Unique identifier of this bus.
    pub id: RouteNodeId,
    /// Human-readable name.
    pub name: String,
    /// Signal kind this bus carries.
    pub signal_kind: SignalKind,
    /// Whether the bus is soloed.
    pub solo: bool,
    /// Whether the bus is muted.
    pub mute: bool,
}

impl RouteBus {
    /// Create a new routing bus.
    #[must_use]
    pub fn new(id: RouteNodeId, name: impl Into<String>, signal_kind: SignalKind) -> Self {
        Self {
            id,
            name: name.into(),
            signal_kind,
            solo: false,
            mute: false,
        }
    }
}

/// The routing matrix that manages all connections between timeline tracks.
#[derive(Debug, Clone)]
pub struct TrackRoutingMatrix {
    /// All routes indexed by their id.
    routes: HashMap<u64, Route>,
    /// Buses available in the matrix.
    buses: HashMap<RouteNodeId, RouteBus>,
    /// Next route id counter.
    next_route_id: u64,
    /// Next node id counter.
    next_node_id: u64,
}

impl Default for TrackRoutingMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackRoutingMatrix {
    /// Create an empty routing matrix.
    #[must_use]
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
            buses: HashMap::new(),
            next_route_id: 1,
            next_node_id: 1,
        }
    }

    /// Allocate a new unique node id.
    pub fn allocate_node_id(&mut self) -> RouteNodeId {
        let id = RouteNodeId::new(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    /// Add a bus to the routing matrix.
    pub fn add_bus(&mut self, name: impl Into<String>, signal_kind: SignalKind) -> RouteNodeId {
        let id = self.allocate_node_id();
        let bus = RouteBus::new(id, name, signal_kind);
        self.buses.insert(id, bus);
        id
    }

    /// Remove a bus and all routes connected to it.
    pub fn remove_bus(&mut self, bus_id: RouteNodeId) -> bool {
        if self.buses.remove(&bus_id).is_some() {
            self.routes
                .retain(|_, r| r.source != bus_id && r.destination != bus_id);
            true
        } else {
            false
        }
    }

    /// Get a reference to a bus by id.
    #[must_use]
    pub fn get_bus(&self, bus_id: RouteNodeId) -> Option<&RouteBus> {
        self.buses.get(&bus_id)
    }

    /// Return the number of buses.
    #[must_use]
    pub fn bus_count(&self) -> usize {
        self.buses.len()
    }

    /// Connect a source node to a destination node.
    pub fn connect(
        &mut self,
        source: RouteNodeId,
        destination: RouteNodeId,
        signal_kind: SignalKind,
    ) -> u64 {
        let id = self.next_route_id;
        self.next_route_id += 1;
        let route = Route::new(id, source, destination, signal_kind);
        self.routes.insert(id, route);
        id
    }

    /// Disconnect (remove) a route by its id.
    pub fn disconnect(&mut self, route_id: u64) -> bool {
        self.routes.remove(&route_id).is_some()
    }

    /// Set gain on an existing route.
    pub fn set_gain(&mut self, route_id: u64, gain: f64) -> bool {
        if let Some(route) = self.routes.get_mut(&route_id) {
            route.gain = gain.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Enable or disable a route.
    pub fn set_enabled(&mut self, route_id: u64, enabled: bool) -> bool {
        if let Some(route) = self.routes.get_mut(&route_id) {
            route.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Return all routes originating from a given source node.
    #[must_use]
    pub fn routes_from(&self, source: RouteNodeId) -> Vec<&Route> {
        self.routes
            .values()
            .filter(|r| r.source == source)
            .collect()
    }

    /// Return all routes arriving at a given destination node.
    #[must_use]
    pub fn routes_to(&self, destination: RouteNodeId) -> Vec<&Route> {
        self.routes
            .values()
            .filter(|r| r.destination == destination)
            .collect()
    }

    /// Return the total number of routes in the matrix.
    #[must_use]
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// Check whether a route between two nodes already exists.
    #[must_use]
    pub fn is_connected(&self, source: RouteNodeId, destination: RouteNodeId) -> bool {
        self.routes
            .values()
            .any(|r| r.source == source && r.destination == destination)
    }

    /// Detect simple cycles: returns true if adding a route from `source` to `destination`
    /// would create a direct back-edge (destination already routes to source).
    #[must_use]
    pub fn would_create_cycle(&self, source: RouteNodeId, destination: RouteNodeId) -> bool {
        // Simple direct cycle detection
        if source == destination {
            return true;
        }
        self.routes
            .values()
            .any(|r| r.source == destination && r.destination == source && r.enabled)
    }

    /// Clear all routes and buses.
    pub fn clear(&mut self) {
        self.routes.clear();
        self.buses.clear();
    }

    /// Compute the effective gain for a path from source to destination (direct route only).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn effective_gain(&self, source: RouteNodeId, destination: RouteNodeId) -> f64 {
        let gains: Vec<f64> = self
            .routes
            .values()
            .filter(|r| r.source == source && r.destination == destination && r.enabled)
            .map(|r| r.gain)
            .collect();
        if gains.is_empty() {
            0.0
        } else {
            // Sum gains from parallel routes
            gains.iter().sum()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_node_id_creation() {
        let id = RouteNodeId::new(42);
        assert_eq!(id.value(), 42);
    }

    #[test]
    fn test_route_creation() {
        let src = RouteNodeId::new(1);
        let dst = RouteNodeId::new(2);
        let route = Route::new(1, src, dst, SignalKind::AudioStereo);
        assert_eq!(route.gain, 1.0);
        assert!(route.enabled);
        assert!(route.label.is_none());
    }

    #[test]
    fn test_route_with_gain() {
        let src = RouteNodeId::new(1);
        let dst = RouteNodeId::new(2);
        let route = Route::new(1, src, dst, SignalKind::AudioMono).with_gain(0.5);
        assert!((route.gain - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_route_gain_clamping() {
        let src = RouteNodeId::new(1);
        let dst = RouteNodeId::new(2);
        let route = Route::new(1, src, dst, SignalKind::Video).with_gain(2.0);
        assert!((route.gain - 1.0).abs() < f64::EPSILON);
        let route2 = Route::new(2, src, dst, SignalKind::Video).with_gain(-1.0);
        assert!((route2.gain).abs() < f64::EPSILON);
    }

    #[test]
    fn test_route_with_label() {
        let src = RouteNodeId::new(1);
        let dst = RouteNodeId::new(2);
        let route = Route::new(1, src, dst, SignalKind::Data).with_label("Main Send");
        assert_eq!(route.label.as_deref(), Some("Main Send"));
    }

    #[test]
    fn test_matrix_add_bus() {
        let mut matrix = TrackRoutingMatrix::new();
        let bus_id = matrix.add_bus("Master Bus", SignalKind::AudioStereo);
        assert_eq!(matrix.bus_count(), 1);
        let bus = matrix.get_bus(bus_id).expect("should succeed in test");
        assert_eq!(bus.name, "Master Bus");
        assert!(!bus.solo);
        assert!(!bus.mute);
    }

    #[test]
    fn test_matrix_remove_bus_cleans_routes() {
        let mut matrix = TrackRoutingMatrix::new();
        let bus_id = matrix.add_bus("Bus A", SignalKind::AudioMono);
        let other = matrix.allocate_node_id();
        matrix.connect(other, bus_id, SignalKind::AudioMono);
        assert_eq!(matrix.route_count(), 1);
        assert!(matrix.remove_bus(bus_id));
        assert_eq!(matrix.route_count(), 0);
        assert_eq!(matrix.bus_count(), 0);
    }

    #[test]
    fn test_matrix_connect_disconnect() {
        let mut matrix = TrackRoutingMatrix::new();
        let src = matrix.allocate_node_id();
        let dst = matrix.allocate_node_id();
        let route_id = matrix.connect(src, dst, SignalKind::AudioStereo);
        assert_eq!(matrix.route_count(), 1);
        assert!(matrix.is_connected(src, dst));
        assert!(matrix.disconnect(route_id));
        assert_eq!(matrix.route_count(), 0);
        assert!(!matrix.is_connected(src, dst));
    }

    #[test]
    fn test_matrix_set_gain() {
        let mut matrix = TrackRoutingMatrix::new();
        let src = matrix.allocate_node_id();
        let dst = matrix.allocate_node_id();
        let route_id = matrix.connect(src, dst, SignalKind::Video);
        assert!(matrix.set_gain(route_id, 0.75));
        let routes = matrix.routes_from(src);
        assert_eq!(routes.len(), 1);
        assert!((routes[0].gain - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_matrix_set_enabled() {
        let mut matrix = TrackRoutingMatrix::new();
        let src = matrix.allocate_node_id();
        let dst = matrix.allocate_node_id();
        let route_id = matrix.connect(src, dst, SignalKind::AudioStereo);
        assert!(matrix.set_enabled(route_id, false));
        let routes = matrix.routes_from(src);
        assert!(!routes[0].enabled);
    }

    #[test]
    fn test_matrix_routes_to() {
        let mut matrix = TrackRoutingMatrix::new();
        let a = matrix.allocate_node_id();
        let b = matrix.allocate_node_id();
        let c = matrix.allocate_node_id();
        matrix.connect(a, c, SignalKind::AudioMono);
        matrix.connect(b, c, SignalKind::AudioMono);
        let to_c = matrix.routes_to(c);
        assert_eq!(to_c.len(), 2);
    }

    #[test]
    fn test_would_create_cycle() {
        let mut matrix = TrackRoutingMatrix::new();
        let a = matrix.allocate_node_id();
        let b = matrix.allocate_node_id();
        matrix.connect(a, b, SignalKind::AudioStereo);
        assert!(matrix.would_create_cycle(b, a));
        assert!(matrix.would_create_cycle(a, a));
        assert!(!matrix.would_create_cycle(a, b));
    }

    #[test]
    fn test_effective_gain() {
        let mut matrix = TrackRoutingMatrix::new();
        let a = matrix.allocate_node_id();
        let b = matrix.allocate_node_id();
        let r1 = matrix.connect(a, b, SignalKind::AudioStereo);
        matrix.set_gain(r1, 0.5);
        let _r2 = matrix.connect(a, b, SignalKind::AudioStereo);
        // r2 has default gain 1.0, so effective = 0.5 + 1.0 = 1.5
        let eff = matrix.effective_gain(a, b);
        assert!((eff - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_matrix_clear() {
        let mut matrix = TrackRoutingMatrix::new();
        matrix.add_bus("Bus", SignalKind::AudioStereo);
        let a = matrix.allocate_node_id();
        let b = matrix.allocate_node_id();
        matrix.connect(a, b, SignalKind::Video);
        matrix.clear();
        assert_eq!(matrix.route_count(), 0);
        assert_eq!(matrix.bus_count(), 0);
    }

    #[test]
    fn test_default_matrix() {
        let matrix = TrackRoutingMatrix::default();
        assert_eq!(matrix.route_count(), 0);
        assert_eq!(matrix.bus_count(), 0);
    }
}
