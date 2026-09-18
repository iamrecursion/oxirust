//! Aux bus management for the production switcher.
//!
//! Manages auxiliary output bus source selection, clean-feed configuration,
//! and aux bus chaining for complex routing scenarios.

#![allow(dead_code)]

use std::collections::HashMap;

/// Error types for aux bus operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxBusError {
    /// The bus index is out of range.
    InvalidBus(usize),
    /// The source index is invalid.
    InvalidSource(usize),
    /// A circular chain was detected.
    CircularChain,
    /// The named clean-feed source does not exist.
    UnknownCleanFeed(String),
}

impl std::fmt::Display for AuxBusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBus(b) => write!(f, "invalid aux bus: {b}"),
            Self::InvalidSource(s) => write!(f, "invalid source: {s}"),
            Self::CircularChain => write!(f, "circular chain detected"),
            Self::UnknownCleanFeed(n) => write!(f, "unknown clean-feed: {n}"),
        }
    }
}

/// A clean-feed configuration: a program output with a specific downstream
/// keyer or graphic layer removed.
#[derive(Debug, Clone)]
pub struct CleanFeedConfig {
    /// Name of this clean-feed configuration.
    pub name: String,
    /// Base source (e.g. program output index).
    pub base_source: usize,
    /// Downstream keyer layers to remove (by index).
    pub remove_keyers: Vec<usize>,
    /// Whether to strip embedded graphics.
    pub strip_graphics: bool,
}

impl CleanFeedConfig {
    /// Create a new clean-feed configuration.
    #[must_use]
    pub fn new(name: impl Into<String>, base_source: usize) -> Self {
        Self {
            name: name.into(),
            base_source,
            remove_keyers: Vec::new(),
            strip_graphics: false,
        }
    }

    /// Add a keyer layer to remove.
    pub fn remove_keyer(mut self, keyer: usize) -> Self {
        self.remove_keyers.push(keyer);
        self
    }

    /// Enable graphics stripping.
    pub fn strip_graphics(mut self) -> Self {
        self.strip_graphics = true;
        self
    }

    /// Returns true if any modification is applied to the base source.
    #[must_use]
    pub fn has_modifications(&self) -> bool {
        !self.remove_keyers.is_empty() || self.strip_graphics
    }
}

/// The source type for an aux bus output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxSource {
    /// Physical input source by index.
    Input(usize),
    /// Program output of an M/E row.
    Program(usize),
    /// Preview output of an M/E row.
    Preview(usize),
    /// Clean-feed output by name.
    CleanFeed(String),
    /// Black / silence.
    Black,
    /// Color bars test signal.
    Bars,
    /// Another aux bus (chaining).
    AuxChain(usize),
}

impl AuxSource {
    /// Returns true if this source refers to another aux bus.
    #[must_use]
    pub fn is_chain(&self) -> bool {
        matches!(self, Self::AuxChain(_))
    }
}

/// A single aux bus.
#[derive(Debug, Clone)]
pub struct AuxBus {
    /// Bus index (0-based).
    pub index: usize,
    /// Display label.
    pub label: String,
    /// Current source.
    pub source: AuxSource,
    /// Whether this bus is enabled.
    pub enabled: bool,
    /// Optional description.
    pub description: Option<String>,
}

impl AuxBus {
    /// Create a new aux bus defaulting to Black.
    #[must_use]
    pub fn new(index: usize, label: impl Into<String>) -> Self {
        Self {
            index,
            label: label.into(),
            source: AuxSource::Black,
            enabled: true,
            description: None,
        }
    }

    /// Set the source.
    pub fn set_source(&mut self, source: AuxSource) {
        self.source = source;
    }

    /// Set the description.
    pub fn set_description(&mut self, desc: impl Into<String>) {
        self.description = Some(desc.into());
    }
}

/// Manager for all aux buses in the switcher.
#[derive(Debug)]
pub struct AuxBusManager {
    /// Ordered list of aux buses.
    buses: Vec<AuxBus>,
    /// Registered clean-feed configurations, keyed by name.
    clean_feeds: HashMap<String, CleanFeedConfig>,
    /// Maximum number of buses.
    max_buses: usize,
}

impl AuxBusManager {
    /// Create a new manager with a fixed number of buses.
    #[must_use]
    pub fn new(num_buses: usize) -> Self {
        let buses = (0..num_buses)
            .map(|i| AuxBus::new(i, format!("AUX {}", i + 1)))
            .collect();
        Self {
            buses,
            clean_feeds: HashMap::new(),
            max_buses: num_buses,
        }
    }

    /// Number of buses.
    #[must_use]
    pub fn bus_count(&self) -> usize {
        self.buses.len()
    }

    /// Get an aux bus by index.
    #[must_use]
    pub fn bus(&self, index: usize) -> Option<&AuxBus> {
        self.buses.get(index)
    }

    /// Get an aux bus mutably by index.
    #[must_use]
    pub fn bus_mut(&mut self, index: usize) -> Option<&mut AuxBus> {
        self.buses.get_mut(index)
    }

    /// Set the source of an aux bus.
    pub fn set_source(&mut self, bus_index: usize, source: AuxSource) -> Result<(), AuxBusError> {
        if bus_index >= self.buses.len() {
            return Err(AuxBusError::InvalidBus(bus_index));
        }
        // Validate clean-feed reference
        if let AuxSource::CleanFeed(ref name) = source {
            if !self.clean_feeds.contains_key(name) {
                return Err(AuxBusError::UnknownCleanFeed(name.clone()));
            }
        }
        // Detect direct circular chain (self-reference)
        if let AuxSource::AuxChain(target) = source {
            if target == bus_index {
                return Err(AuxBusError::CircularChain);
            }
        }
        self.buses[bus_index].source = source;
        Ok(())
    }

    /// Register a clean-feed configuration.
    pub fn register_clean_feed(&mut self, config: CleanFeedConfig) {
        self.clean_feeds.insert(config.name.clone(), config);
    }

    /// Get a clean-feed configuration by name.
    #[must_use]
    pub fn clean_feed(&self, name: &str) -> Option<&CleanFeedConfig> {
        self.clean_feeds.get(name)
    }

    /// Resolve the ultimate source of a bus, following chains up to a depth
    /// limit to prevent infinite loops.
    #[must_use]
    pub fn resolve_source(&self, bus_index: usize) -> Option<&AuxSource> {
        const MAX_DEPTH: usize = 16;
        let mut current = bus_index;
        for _ in 0..MAX_DEPTH {
            let bus = self.buses.get(current)?;
            if let AuxSource::AuxChain(next) = bus.source {
                current = next;
            } else {
                return Some(&self.buses[current].source);
            }
        }
        None // circular or too deep
    }

    /// Enable or disable a bus.
    pub fn set_enabled(&mut self, bus_index: usize, enabled: bool) -> Result<(), AuxBusError> {
        let bus = self
            .buses
            .get_mut(bus_index)
            .ok_or(AuxBusError::InvalidBus(bus_index))?;
        bus.enabled = enabled;
        Ok(())
    }

    /// Get all buses that are currently outputting the given source.
    #[must_use]
    pub fn buses_with_source(&self, source: &AuxSource) -> Vec<usize> {
        self.buses
            .iter()
            .filter(|b| &b.source == source)
            .map(|b| b.index)
            .collect()
    }

    /// Count how many buses are enabled.
    #[must_use]
    pub fn enabled_count(&self) -> usize {
        self.buses.iter().filter(|b| b.enabled).count()
    }

    /// Swap the sources of two buses.
    pub fn swap_sources(&mut self, a: usize, b: usize) -> Result<(), AuxBusError> {
        if a >= self.buses.len() {
            return Err(AuxBusError::InvalidBus(a));
        }
        if b >= self.buses.len() {
            return Err(AuxBusError::InvalidBus(b));
        }
        let src_a = self.buses[a].source.clone();
        let src_b = self.buses[b].source.clone();
        self.buses[a].source = src_b;
        self.buses[b].source = src_a;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager(n: usize) -> AuxBusManager {
        AuxBusManager::new(n)
    }

    #[test]
    fn test_manager_creation() {
        let mgr = make_manager(4);
        assert_eq!(mgr.bus_count(), 4);
    }

    #[test]
    fn test_bus_default_source_is_black() {
        let mgr = make_manager(2);
        assert_eq!(
            mgr.bus(0).expect("should succeed in test").source,
            AuxSource::Black
        );
    }

    #[test]
    fn test_set_source_input() {
        let mut mgr = make_manager(4);
        mgr.set_source(0, AuxSource::Input(3))
            .expect("should succeed in test");
        assert_eq!(
            mgr.bus(0).expect("should succeed in test").source,
            AuxSource::Input(3)
        );
    }

    #[test]
    fn test_set_source_invalid_bus_errors() {
        let mut mgr = make_manager(4);
        let err = mgr.set_source(99, AuxSource::Black);
        assert_eq!(err, Err(AuxBusError::InvalidBus(99)));
    }

    #[test]
    fn test_set_source_self_chain_errors() {
        let mut mgr = make_manager(4);
        let err = mgr.set_source(2, AuxSource::AuxChain(2));
        assert_eq!(err, Err(AuxBusError::CircularChain));
    }

    #[test]
    fn test_set_source_unknown_clean_feed_errors() {
        let mut mgr = make_manager(4);
        let err = mgr.set_source(0, AuxSource::CleanFeed("CF1".into()));
        assert_eq!(err, Err(AuxBusError::UnknownCleanFeed("CF1".into())));
    }

    #[test]
    fn test_register_and_use_clean_feed() {
        let mut mgr = make_manager(4);
        let cfg = CleanFeedConfig::new("CF1", 0)
            .remove_keyer(0)
            .strip_graphics();
        mgr.register_clean_feed(cfg);
        assert!(mgr
            .set_source(1, AuxSource::CleanFeed("CF1".into()))
            .is_ok());
        assert_eq!(
            mgr.bus(1).expect("should succeed in test").source,
            AuxSource::CleanFeed("CF1".into())
        );
    }

    #[test]
    fn test_resolve_source_direct() {
        let mut mgr = make_manager(4);
        mgr.set_source(0, AuxSource::Program(0))
            .expect("should succeed in test");
        let src = mgr.resolve_source(0).expect("should succeed in test");
        assert_eq!(src, &AuxSource::Program(0));
    }

    #[test]
    fn test_resolve_source_chain() {
        let mut mgr = make_manager(4);
        mgr.set_source(0, AuxSource::Input(5))
            .expect("should succeed in test");
        mgr.set_source(1, AuxSource::AuxChain(0))
            .expect("should succeed in test");
        let src = mgr.resolve_source(1).expect("should succeed in test");
        assert_eq!(src, &AuxSource::Input(5));
    }

    #[test]
    fn test_enable_disable_bus() {
        let mut mgr = make_manager(4);
        mgr.set_enabled(2, false).expect("should succeed in test");
        assert!(!mgr.bus(2).expect("should succeed in test").enabled);
        assert_eq!(mgr.enabled_count(), 3);
    }

    #[test]
    fn test_set_enabled_invalid_bus_errors() {
        let mut mgr = make_manager(4);
        assert_eq!(mgr.set_enabled(99, false), Err(AuxBusError::InvalidBus(99)));
    }

    #[test]
    fn test_buses_with_source() {
        let mut mgr = make_manager(4);
        mgr.set_source(1, AuxSource::Bars)
            .expect("should succeed in test");
        mgr.set_source(3, AuxSource::Bars)
            .expect("should succeed in test");
        let buses = mgr.buses_with_source(&AuxSource::Bars);
        assert_eq!(buses.len(), 2);
        assert!(buses.contains(&1));
        assert!(buses.contains(&3));
    }

    #[test]
    fn test_swap_sources() {
        let mut mgr = make_manager(4);
        mgr.set_source(0, AuxSource::Input(1))
            .expect("should succeed in test");
        mgr.set_source(1, AuxSource::Program(0))
            .expect("should succeed in test");
        mgr.swap_sources(0, 1).expect("should succeed in test");
        assert_eq!(
            mgr.bus(0).expect("should succeed in test").source,
            AuxSource::Program(0)
        );
        assert_eq!(
            mgr.bus(1).expect("should succeed in test").source,
            AuxSource::Input(1)
        );
    }

    #[test]
    fn test_clean_feed_has_modifications() {
        let cfg = CleanFeedConfig::new("CF", 0).remove_keyer(1);
        assert!(cfg.has_modifications());
        let cfg2 = CleanFeedConfig::new("CF2", 0);
        assert!(!cfg2.has_modifications());
    }

    #[test]
    fn test_aux_source_is_chain() {
        assert!(AuxSource::AuxChain(0).is_chain());
        assert!(!AuxSource::Input(0).is_chain());
    }
}
