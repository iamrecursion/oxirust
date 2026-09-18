//! Output routing for the playout server.
//!
//! `OutputRouter` manages a set of named outputs and decides which sources are
//! sent to which destinations based on a routing table.  Supports primary/
//! backup pairs for fail-safe routing.

#![allow(dead_code)]

use std::collections::HashMap;

/// The type of a playout output destination.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OutputKind {
    /// Serial digital interface (SDI) output.
    Sdi,
    /// NDI network output.
    Ndi,
    /// RTMP stream to a CDN or streaming platform.
    Rtmp,
    /// SRT (Secure Reliable Transport) stream.
    Srt,
    /// IP multicast (e.g. SMPTE ST 2110 / RTP).
    IpMulticast,
    /// File recording output.
    FileRecord,
}

impl OutputKind {
    /// Returns `true` for outputs that carry a live network feed.
    #[must_use]
    pub fn is_network(self: &OutputKind) -> bool {
        matches!(self, Self::Ndi | Self::Rtmp | Self::Srt | Self::IpMulticast)
    }
}

/// Status of a single output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputStatus {
    /// Output is connected and passing signal.
    Active,
    /// Output is configured but not yet started.
    Inactive,
    /// Output has faulted and is not passing signal.
    Faulted,
    /// Output is disabled by the operator.
    Disabled,
}

impl OutputStatus {
    /// Returns `true` if the output is currently passing signal.
    #[must_use]
    pub fn is_passing(self) -> bool {
        self == Self::Active
    }
}

/// A single output destination.
#[derive(Debug, Clone)]
pub struct OutputDestination {
    /// Unique name for this output (e.g. "SDI-OUT-1").
    pub name: String,
    /// Type of output.
    pub kind: OutputKind,
    /// Current status.
    pub status: OutputStatus,
    /// Address or path (e.g. URL, device path).
    pub address: String,
    /// Optional backup destination name.
    pub backup: Option<String>,
}

impl OutputDestination {
    /// Create a new output destination in `Inactive` state.
    #[must_use]
    pub fn new(name: impl Into<String>, kind: OutputKind, address: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind,
            status: OutputStatus::Inactive,
            address: address.into(),
            backup: None,
        }
    }

    /// Attach a named backup destination.
    pub fn with_backup(mut self, backup: impl Into<String>) -> Self {
        self.backup = Some(backup.into());
        self
    }

    /// Returns `true` if a backup is configured.
    #[must_use]
    pub fn has_backup(&self) -> bool {
        self.backup.is_some()
    }
}

/// Route entry: maps a source identifier to an output destination name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    /// Source identifier (e.g. "PGM", "VT1").
    pub source: String,
    /// Name of the output destination.
    pub destination: String,
}

impl Route {
    #[must_use]
    pub fn new(source: impl Into<String>, destination: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            destination: destination.into(),
        }
    }
}

/// Manages all output destinations and their routing.
#[derive(Debug, Default)]
pub struct OutputRouter {
    destinations: HashMap<String, OutputDestination>,
    routes: Vec<Route>,
}

impl OutputRouter {
    /// Create an empty router.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a destination.  Overwrites any existing destination with the
    /// same name.
    pub fn add_destination(&mut self, dest: OutputDestination) {
        self.destinations.insert(dest.name.clone(), dest);
    }

    /// Remove a destination and all routes pointing to it.
    pub fn remove_destination(&mut self, name: &str) -> bool {
        if self.destinations.remove(name).is_none() {
            return false;
        }
        self.routes.retain(|r| r.destination != name);
        true
    }

    /// Look up a destination by name.
    #[must_use]
    pub fn destination(&self, name: &str) -> Option<&OutputDestination> {
        self.destinations.get(name)
    }

    /// Update the status of a destination.  Returns `false` if not found.
    pub fn set_status(&mut self, name: &str, status: OutputStatus) -> bool {
        if let Some(dest) = self.destinations.get_mut(name) {
            dest.status = status;
            true
        } else {
            false
        }
    }

    /// Add a route from source to destination.  If a route for the same
    /// source already exists it is replaced.
    pub fn route(&mut self, source: impl Into<String>, destination: impl Into<String>) {
        let src = source.into();
        self.routes.retain(|r| r.source != src);
        self.routes.push(Route::new(src, destination));
    }

    /// Find the destination name for a given source.
    #[must_use]
    pub fn resolve(&self, source: &str) -> Option<&str> {
        self.routes
            .iter()
            .find(|r| r.source == source)
            .map(|r| r.destination.as_str())
    }

    /// Resolve the destination object for a given source, following to backup
    /// if the primary is faulted.
    #[must_use]
    pub fn resolve_with_failover(&self, source: &str) -> Option<&OutputDestination> {
        let dest_name = self.resolve(source)?;
        let dest = self.destinations.get(dest_name)?;
        if dest.status == OutputStatus::Faulted {
            if let Some(backup_name) = &dest.backup {
                return self.destinations.get(backup_name);
            }
        }
        Some(dest)
    }

    /// All active destinations.
    #[must_use]
    pub fn active_destinations(&self) -> Vec<&OutputDestination> {
        self.destinations
            .values()
            .filter(|d| d.status == OutputStatus::Active)
            .collect()
    }

    /// Number of destinations registered.
    #[must_use]
    pub fn destination_count(&self) -> usize {
        self.destinations.len()
    }

    /// Number of routes configured.
    #[must_use]
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sdi_dest(name: &str) -> OutputDestination {
        OutputDestination::new(name, OutputKind::Sdi, "/dev/sdi0")
    }

    #[test]
    fn output_kind_is_network() {
        assert!(OutputKind::Rtmp.is_network());
        assert!(OutputKind::Ndi.is_network());
        assert!(!OutputKind::Sdi.is_network());
        assert!(!OutputKind::FileRecord.is_network());
    }

    #[test]
    fn output_status_is_passing() {
        assert!(OutputStatus::Active.is_passing());
        assert!(!OutputStatus::Faulted.is_passing());
        assert!(!OutputStatus::Inactive.is_passing());
    }

    #[test]
    fn new_destination_is_inactive() {
        let d = sdi_dest("SDI-1");
        assert_eq!(d.status, OutputStatus::Inactive);
    }

    #[test]
    fn destination_with_backup() {
        let d = sdi_dest("SDI-1").with_backup("SDI-2");
        assert!(d.has_backup());
        assert_eq!(d.backup.as_deref(), Some("SDI-2"));
    }

    #[test]
    fn add_and_look_up_destination() {
        let mut router = OutputRouter::new();
        router.add_destination(sdi_dest("SDI-1"));
        assert!(router.destination("SDI-1").is_some());
        assert_eq!(router.destination_count(), 1);
    }

    #[test]
    fn remove_destination_removes_routes() {
        let mut router = OutputRouter::new();
        router.add_destination(sdi_dest("SDI-1"));
        router.route("PGM", "SDI-1");
        assert!(router.remove_destination("SDI-1"));
        assert_eq!(router.destination_count(), 0);
        assert_eq!(router.route_count(), 0);
    }

    #[test]
    fn remove_nonexistent_destination_returns_false() {
        let mut router = OutputRouter::new();
        assert!(!router.remove_destination("GHOST"));
    }

    #[test]
    fn set_status_updates_destination() {
        let mut router = OutputRouter::new();
        router.add_destination(sdi_dest("SDI-1"));
        assert!(router.set_status("SDI-1", OutputStatus::Active));
        assert_eq!(
            router
                .destination("SDI-1")
                .expect("should succeed in test")
                .status,
            OutputStatus::Active
        );
    }

    #[test]
    fn resolve_returns_destination_name() {
        let mut router = OutputRouter::new();
        router.add_destination(sdi_dest("SDI-1"));
        router.route("PGM", "SDI-1");
        assert_eq!(router.resolve("PGM"), Some("SDI-1"));
    }

    #[test]
    fn resolve_unrouted_source_returns_none() {
        let router = OutputRouter::new();
        assert!(router.resolve("PGM").is_none());
    }

    #[test]
    fn resolve_with_failover_returns_backup_when_primary_faulted() {
        let mut router = OutputRouter::new();
        let primary = sdi_dest("SDI-1").with_backup("SDI-2");
        router.add_destination(primary);
        router.add_destination(sdi_dest("SDI-2"));
        router.set_status("SDI-1", OutputStatus::Faulted);
        router.set_status("SDI-2", OutputStatus::Active);
        router.route("PGM", "SDI-1");
        let resolved = router
            .resolve_with_failover("PGM")
            .expect("should succeed in test");
        assert_eq!(resolved.name, "SDI-2");
    }

    #[test]
    fn active_destinations_list() {
        let mut router = OutputRouter::new();
        router.add_destination(sdi_dest("SDI-1"));
        router.add_destination(sdi_dest("SDI-2"));
        router.set_status("SDI-1", OutputStatus::Active);
        let active = router.active_destinations();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn route_replaces_existing_for_same_source() {
        let mut router = OutputRouter::new();
        router.add_destination(sdi_dest("SDI-1"));
        router.add_destination(sdi_dest("SDI-2"));
        router.route("PGM", "SDI-1");
        router.route("PGM", "SDI-2");
        assert_eq!(router.route_count(), 1);
        assert_eq!(router.resolve("PGM"), Some("SDI-2"));
    }
}
