//! Multi-site automation coordination across geographically distributed facilities.
//!
//! This module provides a [`SiteRegistry`] that tracks the set of remote
//! automation sites, their reachability, and a simple command broadcast
//! mechanism for issuing synchronised control actions to multiple facilities
//! simultaneously.
//!
//! # Concepts
//!
//! - **Site**: A remote broadcast facility identified by a unique name and
//!   a network endpoint (address + port).
//! - **SiteCommand**: A control action dispatched to one or more sites.
//! - **SiteStatus**: Availability state of a remote site (online/offline/
//!   degraded).
//! - **SiteRegistry**: Central coordinator that tracks sites, evaluates their
//!   health, and records dispatched commands.
//!
//! In a real deployment each site would be connected via a low-latency WAN
//! link; this module deliberately avoids actual network I/O so it can be
//! tested deterministically.  Networking adapters wrap this registry.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

// ─────────────────────────────────────────────────────────────────────────────
// Site descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// Connectivity status of a remote site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SiteStatus {
    /// Full connectivity; all channels operational.
    Online,
    /// Site is reachable but experiencing issues (partial service).
    Degraded,
    /// Site is not reachable.
    Offline,
}

impl SiteStatus {
    /// Returns `true` if the site can accept commands.
    pub fn is_reachable(self) -> bool {
        !matches!(self, Self::Offline)
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Online => "online",
            Self::Degraded => "degraded",
            Self::Offline => "offline",
        }
    }
}

/// Description of a remote broadcast facility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Site {
    /// Unique site identifier (e.g. `"NYC-1"`, `"LON-2"`).
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Hostname or IP address of the site's automation endpoint.
    pub address: String,
    /// TCP port for the site's control API.
    pub port: u16,
    /// Current connectivity status.
    pub status: SiteStatus,
    /// Number of channels managed by this site.
    pub channel_count: usize,
    /// Millisecond timestamp of the most recent successful health check.
    pub last_seen_ms: u64,
    /// Geographic region code (e.g. `"US-EAST"`, `"EU-WEST"`).
    pub region: String,
}

impl Site {
    /// Create a new site descriptor.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        address: impl Into<String>,
        port: u16,
        region: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            address: address.into(),
            port,
            status: SiteStatus::Offline,
            channel_count: 0,
            last_seen_ms: 0,
            region: region.into(),
        }
    }

    /// Mark the site as online, updating the last-seen timestamp.
    pub fn mark_online(&mut self, now_ms: u64) {
        self.status = SiteStatus::Online;
        self.last_seen_ms = now_ms;
    }

    /// Mark the site as offline.
    pub fn mark_offline(&mut self) {
        self.status = SiteStatus::Offline;
    }

    /// Mark the site as degraded.
    pub fn mark_degraded(&mut self, now_ms: u64) {
        self.status = SiteStatus::Degraded;
        self.last_seen_ms = now_ms;
    }

    /// Returns `true` if the site has not reported in within `timeout_ms`.
    pub fn is_stale(&self, now_ms: u64, timeout_ms: u64) -> bool {
        now_ms.saturating_sub(self.last_seen_ms) > timeout_ms
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Site command
// ─────────────────────────────────────────────────────────────────────────────

/// A control command dispatched to one or more remote sites.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteCommand {
    /// Unique command identifier.
    pub id: String,
    /// Command verb (e.g. `"take"`, `"cut"`, `"hold"`, `"start_channel"`).
    pub verb: String,
    /// Target site IDs.  An empty list means "broadcast to all reachable sites".
    pub targets: Vec<String>,
    /// Optional JSON-serialisable payload.
    pub payload: Option<serde_json::Value>,
    /// Millisecond timestamp when the command was issued.
    pub issued_at_ms: u64,
}

impl SiteCommand {
    /// Create a new site command targeting specific sites.
    pub fn new(
        id: impl Into<String>,
        verb: impl Into<String>,
        targets: Vec<String>,
        issued_at_ms: u64,
    ) -> Self {
        Self {
            id: id.into(),
            verb: verb.into(),
            targets,
            payload: None,
            issued_at_ms,
        }
    }

    /// Create a broadcast command (targets all reachable sites).
    pub fn broadcast(id: impl Into<String>, verb: impl Into<String>, issued_at_ms: u64) -> Self {
        Self::new(id, verb, Vec::new(), issued_at_ms)
    }

    /// Attach a JSON payload.
    pub fn with_payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = Some(payload);
        self
    }
}

/// Record of a dispatched command including which sites actually received it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchRecord {
    /// The command that was dispatched.
    pub command: SiteCommand,
    /// Site IDs that received the command.
    pub delivered_to: Vec<String>,
    /// Site IDs that were skipped (offline or unreachable).
    pub skipped: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Registry
// ─────────────────────────────────────────────────────────────────────────────

/// Central registry for multi-site broadcast automation coordination.
#[derive(Debug, Default)]
pub struct SiteRegistry {
    /// Registered sites keyed by site ID.
    sites: HashMap<String, Site>,
    /// Command dispatch history.
    dispatch_log: Vec<DispatchRecord>,
}

impl SiteRegistry {
    /// Create an empty site registry.
    pub fn new() -> Self {
        Self::default()
    }

    // ── Site management ───────────────────────────────────────────────────────

    /// Register or replace a site.
    pub fn register_site(&mut self, site: Site) {
        info!(
            "Registered site '{}' ({}) at {}:{} region={}",
            site.id, site.name, site.address, site.port, site.region
        );
        self.sites.insert(site.id.clone(), site);
    }

    /// Remove a site from the registry.  Returns the removed site, if found.
    pub fn remove_site(&mut self, id: &str) -> Option<Site> {
        self.sites.remove(id)
    }

    /// Get an immutable reference to a site.
    pub fn get_site(&self, id: &str) -> Option<&Site> {
        self.sites.get(id)
    }

    /// Get a mutable reference to a site.
    pub fn get_site_mut(&mut self, id: &str) -> Option<&mut Site> {
        self.sites.get_mut(id)
    }

    /// Return all sites that are currently reachable (Online or Degraded).
    pub fn reachable_sites(&self) -> Vec<&Site> {
        self.sites
            .values()
            .filter(|s| s.status.is_reachable())
            .collect()
    }

    /// Return all sites in a given region.
    pub fn sites_in_region(&self, region: &str) -> Vec<&Site> {
        self.sites.values().filter(|s| s.region == region).collect()
    }

    /// Total number of registered sites.
    pub fn site_count(&self) -> usize {
        self.sites.len()
    }

    // ── Health update ─────────────────────────────────────────────────────────

    /// Update the status of a site based on a health check result.
    ///
    /// Marks stale sites (no heartbeat within `stale_timeout_ms`) as
    /// `Offline`.
    pub fn update_health(&mut self, id: &str, now_ms: u64, stale_timeout_ms: u64) {
        if let Some(site) = self.sites.get_mut(id) {
            if site.is_stale(now_ms, stale_timeout_ms) && site.status != SiteStatus::Offline {
                warn!(
                    "Site '{}' is stale (last seen {}ms ago), marking offline",
                    id,
                    now_ms.saturating_sub(site.last_seen_ms)
                );
                site.mark_offline();
            }
        }
    }

    // ── Command dispatch ──────────────────────────────────────────────────────

    /// Dispatch a command to the specified target sites (or all reachable sites
    /// if `command.targets` is empty).
    ///
    /// Returns a [`DispatchRecord`] summarising which sites received the
    /// command and which were skipped due to being unreachable.
    pub fn dispatch(&mut self, command: SiteCommand) -> DispatchRecord {
        // For broadcast (empty targets) we consider ALL registered sites; for targeted
        // commands we only consider the named sites.
        let target_ids: Vec<String> = if command.targets.is_empty() {
            self.sites.keys().cloned().collect()
        } else {
            command.targets.clone()
        };

        let mut delivered_to = Vec::new();
        let mut skipped = Vec::new();

        for site_id in &target_ids {
            match self.sites.get(site_id) {
                Some(site) if site.status.is_reachable() => {
                    info!(
                        "Dispatching '{}' to site '{}' ({})",
                        command.verb,
                        site_id,
                        site.status.label()
                    );
                    delivered_to.push(site_id.clone());
                }
                Some(_site) => {
                    warn!("Skipping site '{}' — offline", site_id);
                    skipped.push(site_id.clone());
                }
                None => {
                    warn!("Unknown site '{}' in command targets", site_id);
                    skipped.push(site_id.clone());
                }
            }
        }

        let record = DispatchRecord {
            command,
            delivered_to,
            skipped,
        };
        self.dispatch_log.push(record.clone());
        record
    }

    /// Return the full command dispatch history.
    pub fn dispatch_log(&self) -> &[DispatchRecord] {
        &self.dispatch_log
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_site(id: &str, region: &str) -> Site {
        Site::new(id, id, "10.0.0.1", 8080, region)
    }

    #[test]
    fn test_register_and_count() {
        let mut reg = SiteRegistry::new();
        reg.register_site(make_site("NYC-1", "US-EAST"));
        reg.register_site(make_site("LON-1", "EU-WEST"));
        assert_eq!(reg.site_count(), 2);
    }

    #[test]
    fn test_reachable_sites_initially_offline() {
        let mut reg = SiteRegistry::new();
        reg.register_site(make_site("s1", "US"));
        assert_eq!(reg.reachable_sites().len(), 0, "sites start offline");
    }

    #[test]
    fn test_mark_online_makes_reachable() {
        let mut reg = SiteRegistry::new();
        reg.register_site(make_site("s2", "EU"));
        reg.get_site_mut("s2")
            .expect("site should exist")
            .mark_online(1000);
        assert_eq!(reg.reachable_sites().len(), 1);
    }

    #[test]
    fn test_region_filter() {
        let mut reg = SiteRegistry::new();
        reg.register_site(make_site("us1", "US-EAST"));
        reg.register_site(make_site("us2", "US-EAST"));
        reg.register_site(make_site("eu1", "EU-WEST"));
        assert_eq!(reg.sites_in_region("US-EAST").len(), 2);
        assert_eq!(reg.sites_in_region("EU-WEST").len(), 1);
    }

    #[test]
    fn test_dispatch_to_online_sites() {
        let mut reg = SiteRegistry::new();
        let mut s1 = make_site("A", "US");
        s1.mark_online(0);
        let mut s2 = make_site("B", "EU");
        s2.mark_online(0);
        reg.register_site(s1);
        reg.register_site(s2);

        let cmd = SiteCommand::broadcast("cmd1", "take", 0);
        let record = reg.dispatch(cmd);
        assert_eq!(record.delivered_to.len(), 2);
        assert!(record.skipped.is_empty());
    }

    #[test]
    fn test_dispatch_skips_offline_sites() {
        let mut reg = SiteRegistry::new();
        let mut online = make_site("online", "US");
        online.mark_online(0);
        let offline = make_site("offline", "US");
        reg.register_site(online);
        reg.register_site(offline);

        let cmd = SiteCommand::broadcast("cmd2", "cut", 0);
        let record = reg.dispatch(cmd);
        assert_eq!(record.delivered_to.len(), 1);
        assert_eq!(record.skipped.len(), 1);
    }

    #[test]
    fn test_dispatch_targeted() {
        let mut reg = SiteRegistry::new();
        let mut s = make_site("target", "US");
        s.mark_online(0);
        reg.register_site(s);

        let cmd = SiteCommand::new("cmd3", "hold", vec!["target".to_string()], 0);
        let record = reg.dispatch(cmd);
        assert_eq!(record.delivered_to, vec!["target".to_string()]);
    }

    #[test]
    fn test_stale_detection() {
        let mut site = make_site("stale", "US");
        site.mark_online(0);
        // 10 000 ms have passed; timeout is 5 000 ms → stale
        assert!(site.is_stale(10_000, 5_000));
    }

    #[test]
    fn test_remove_site() {
        let mut reg = SiteRegistry::new();
        reg.register_site(make_site("del", "US"));
        assert!(reg.remove_site("del").is_some());
        assert_eq!(reg.site_count(), 0);
    }
}
