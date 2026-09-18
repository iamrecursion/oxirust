//! NDI network source registry and discovery helpers
//!
//! This module provides a self-contained, in-memory registry of NDI sources
//! and lightweight discovery helpers that operate without the mDNS daemon
//! (useful for unit testing and embedded scenarios).

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

/// The role that an NDI network entity plays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdiServiceType {
    /// A source that emits audio/video frames.
    Source,
    /// A receiver that consumes frames from a source.
    Receiver,
    /// A bridge that forwards NDI streams between networks.
    Bridge,
}

impl NdiServiceType {
    /// Returns `true` when this entity produces (sends) NDI frames.
    pub fn is_sender(&self) -> bool {
        matches!(self, Self::Source | Self::Bridge)
    }
}

/// Metadata describing a discovered NDI source.
#[derive(Debug, Clone)]
pub struct NdiRegistrySourceInfo {
    /// Human-readable source name (e.g. `"Camera 1"`).
    pub name: String,
    /// Network URL / address string (e.g. `"192.168.1.10:5960"`).
    pub url_address: String,
    /// Machine / host name that is running the source.
    pub machine_name: String,
    /// Groups this source has joined.
    pub groups: Vec<String>,
}

impl NdiRegistrySourceInfo {
    /// Create a new `NdiRegistrySourceInfo`.
    pub fn new(
        name: impl Into<String>,
        url_address: impl Into<String>,
        machine_name: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            url_address: url_address.into(),
            machine_name: machine_name.into(),
            groups: Vec::new(),
        }
    }

    /// Returns the full NDI display name in the canonical `"Machine (Name)"` format.
    pub fn full_name(&self) -> String {
        format!("{} ({})", self.machine_name, self.name)
    }

    /// Returns `true` when the source belongs to the named group.
    pub fn is_in_group(&self, group: &str) -> bool {
        self.groups.iter().any(|g| g == group)
    }
}

/// An in-memory registry of NDI sources discovered on the local network.
#[derive(Debug, Clone)]
pub struct NdiDiscovery {
    /// All currently known sources.
    pub sources: Vec<NdiRegistrySourceInfo>,
    /// Machine name of the local host.
    pub local_machine: String,
}

impl NdiDiscovery {
    /// Create an empty registry.
    pub fn new(local_machine: impl Into<String>) -> Self {
        Self {
            sources: Vec::new(),
            local_machine: local_machine.into(),
        }
    }

    /// Register a new source.
    pub fn add_source(&mut self, source: NdiRegistrySourceInfo) {
        self.sources.push(source);
    }

    /// Remove a source by its URL address.  Returns `true` when a source was removed.
    pub fn remove_source(&mut self, url_address: &str) -> bool {
        let before = self.sources.len();
        self.sources.retain(|s| s.url_address != url_address);
        self.sources.len() < before
    }

    /// Find a source by its display name (case-sensitive exact match).
    pub fn find_by_name(&self, name: &str) -> Option<&NdiRegistrySourceInfo> {
        self.sources.iter().find(|s| s.name == name)
    }

    /// Return all sources that belong to the named group.
    pub fn sources_in_group(&self, group: &str) -> Vec<&NdiRegistrySourceInfo> {
        self.sources
            .iter()
            .filter(|s| s.is_in_group(group))
            .collect()
    }

    /// Return the total number of registered sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Return all sources whose `machine_name` matches `local_machine`.
    pub fn local_sources(&self) -> Vec<&NdiRegistrySourceInfo> {
        self.sources
            .iter()
            .filter(|s| s.machine_name == self.local_machine)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_source(name: &str, machine: &str) -> NdiRegistrySourceInfo {
        NdiRegistrySourceInfo::new(name, format!("{machine}:5960"), machine)
    }

    // --- NdiServiceType ---

    #[test]
    fn test_is_sender_source() {
        assert!(NdiServiceType::Source.is_sender());
    }

    #[test]
    fn test_is_sender_bridge() {
        assert!(NdiServiceType::Bridge.is_sender());
    }

    #[test]
    fn test_is_sender_receiver_false() {
        assert!(!NdiServiceType::Receiver.is_sender());
    }

    // --- NdiRegistrySourceInfo ---

    #[test]
    fn test_full_name_format() {
        let s = make_source("Camera 1", "STUDIO-PC");
        assert_eq!(s.full_name(), "STUDIO-PC (Camera 1)");
    }

    #[test]
    fn test_is_in_group_true() {
        let mut s = make_source("Cam", "PC1");
        s.groups.push("public".to_string());
        assert!(s.is_in_group("public"));
    }

    #[test]
    fn test_is_in_group_false() {
        let s = make_source("Cam", "PC1");
        assert!(!s.is_in_group("private"));
    }

    // --- NdiDiscovery ---

    #[test]
    fn test_add_source_and_count() {
        let mut reg = NdiDiscovery::new("LOCAL");
        reg.add_source(make_source("Cam1", "LOCAL"));
        reg.add_source(make_source("Cam2", "REMOTE"));
        assert_eq!(reg.source_count(), 2);
    }

    #[test]
    fn test_remove_source_found() {
        let mut reg = NdiDiscovery::new("LOCAL");
        let mut s = make_source("Cam1", "LOCAL");
        s.url_address = "192.168.1.5:5960".to_string();
        reg.add_source(s);
        assert!(reg.remove_source("192.168.1.5:5960"));
        assert_eq!(reg.source_count(), 0);
    }

    #[test]
    fn test_remove_source_not_found() {
        let mut reg = NdiDiscovery::new("LOCAL");
        assert!(!reg.remove_source("does-not-exist"));
    }

    #[test]
    fn test_find_by_name_found() {
        let mut reg = NdiDiscovery::new("LOCAL");
        reg.add_source(make_source("Studio Feed", "PC1"));
        assert!(reg.find_by_name("Studio Feed").is_some());
    }

    #[test]
    fn test_find_by_name_not_found() {
        let reg = NdiDiscovery::new("LOCAL");
        assert!(reg.find_by_name("Ghost").is_none());
    }

    #[test]
    fn test_sources_in_group() {
        let mut reg = NdiDiscovery::new("LOCAL");
        let mut s1 = make_source("A", "PC1");
        s1.groups.push("broadcast".to_string());
        let s2 = make_source("B", "PC2");
        reg.add_source(s1);
        reg.add_source(s2);
        assert_eq!(reg.sources_in_group("broadcast").len(), 1);
    }

    #[test]
    fn test_local_sources() {
        let mut reg = NdiDiscovery::new("MY-PC");
        reg.add_source(make_source("Local Cam", "MY-PC"));
        reg.add_source(make_source("Remote Cam", "OTHER-PC"));
        assert_eq!(reg.local_sources().len(), 1);
        assert_eq!(reg.local_sources()[0].name, "Local Cam");
    }

    #[test]
    fn test_source_count_empty() {
        let reg = NdiDiscovery::new("LOCAL");
        assert_eq!(reg.source_count(), 0);
    }

    #[test]
    fn test_local_machine_stored() {
        let reg = NdiDiscovery::new("BROADCAST-PC");
        assert_eq!(reg.local_machine, "BROADCAST-PC");
    }
}
