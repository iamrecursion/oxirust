//! Live ingest management
//!
//! Tracks live ingest sources (SDI, RTMP, SRT, NDI, file, RIST), manages
//! their activation state, and provides aggregate statistics.

#![allow(dead_code)]

/// Transport protocol for a live ingest source
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestProtocol {
    /// Serial Digital Interface (physical baseband)
    Sdi,
    /// Real-Time Messaging Protocol
    Rtmp,
    /// Secure Reliable Transport
    Srt,
    /// Network Device Interface
    Ndi,
    /// Local file (for testing / playout)
    File,
    /// Reliable Internet Stream Transport
    Rist,
}

impl std::fmt::Display for IngestProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sdi => write!(f, "SDI"),
            Self::Rtmp => write!(f, "RTMP"),
            Self::Srt => write!(f, "SRT"),
            Self::Ndi => write!(f, "NDI"),
            Self::File => write!(f, "File"),
            Self::Rist => write!(f, "RIST"),
        }
    }
}

/// A live ingest source
#[derive(Debug, Clone)]
pub struct IngestSource {
    /// Unique identifier
    pub id: u64,
    /// Human-readable name
    pub name: String,
    /// Transport protocol
    pub protocol: IngestProtocol,
    /// Address / URI / device path
    pub address: String,
    /// Whether this source is currently ingesting
    pub active: bool,
    /// Current receive bitrate in kbps (0 when inactive)
    pub bitrate_kbps: u32,
}

impl IngestSource {
    /// Create a new inactive ingest source
    pub fn new(id: u64, name: &str, protocol: IngestProtocol, addr: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            protocol,
            address: addr.to_string(),
            active: false,
            bitrate_kbps: 0,
        }
    }

    /// Whether the source is currently live (active and has measurable bitrate)
    pub fn is_live(&self) -> bool {
        self.active && self.bitrate_kbps > 0
    }
}

/// Manager for all ingest sources
#[derive(Debug, Default)]
pub struct IngestManager {
    sources: Vec<IngestSource>,
    active_count: usize,
}

impl IngestManager {
    /// Create a new empty ingest manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new ingest source
    pub fn add_source(&mut self, src: IngestSource) {
        if src.active {
            self.active_count += 1;
        }
        self.sources.push(src);
    }

    /// Activate a source by id
    ///
    /// # Errors
    ///
    /// Returns an error if no source with `id` exists or it is already active.
    pub fn activate(&mut self, id: u64) -> Result<(), String> {
        let src = self
            .sources
            .iter_mut()
            .find(|s| s.id == id)
            .ok_or_else(|| format!("Source {id} not found"))?;
        if src.active {
            return Err(format!("Source {id} is already active"));
        }
        src.active = true;
        self.active_count += 1;
        Ok(())
    }

    /// Deactivate a source by id
    ///
    /// # Errors
    ///
    /// Returns an error if no source with `id` exists or it is already inactive.
    pub fn deactivate(&mut self, id: u64) -> Result<(), String> {
        let src = self
            .sources
            .iter_mut()
            .find(|s| s.id == id)
            .ok_or_else(|| format!("Source {id} not found"))?;
        if !src.active {
            return Err(format!("Source {id} is already inactive"));
        }
        src.active = false;
        src.bitrate_kbps = 0;
        self.active_count = self.active_count.saturating_sub(1);
        Ok(())
    }

    /// Retrieve all currently active sources
    pub fn active_sources(&self) -> Vec<&IngestSource> {
        self.sources.iter().filter(|s| s.active).collect()
    }

    /// Look up a source by id
    pub fn source_by_id(&self, id: u64) -> Option<&IngestSource> {
        self.sources.iter().find(|s| s.id == id)
    }

    /// Total number of registered sources
    pub fn total_sources(&self) -> usize {
        self.sources.len()
    }
}

/// Aggregate statistics for an `IngestManager`
#[derive(Debug, Clone)]
pub struct IngestStats {
    /// Total registered sources
    pub total_sources: usize,
    /// Currently active sources
    pub active_sources: usize,
    /// Sum of active source bitrates in kbps
    pub total_bitrate_kbps: u32,
}

/// Compute aggregate ingest statistics
pub fn ingest_stats(manager: &IngestManager) -> IngestStats {
    let total_sources = manager.sources.len();
    let active_sources = manager.sources.iter().filter(|s| s.active).count();
    let total_bitrate_kbps = manager
        .sources
        .iter()
        .filter(|s| s.active)
        .map(|s| s.bitrate_kbps)
        .sum();
    IngestStats {
        total_sources,
        active_sources,
        total_bitrate_kbps,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_source(id: u64, proto: IngestProtocol) -> IngestSource {
        IngestSource::new(id, &format!("src_{id}"), proto, "192.168.1.1:1234")
    }

    #[test]
    fn test_source_new_inactive() {
        let s = make_source(1, IngestProtocol::Rtmp);
        assert!(!s.active);
        assert_eq!(s.bitrate_kbps, 0);
    }

    #[test]
    fn test_source_is_live_requires_bitrate() {
        let mut s = make_source(1, IngestProtocol::Rtmp);
        s.active = true;
        assert!(!s.is_live()); // bitrate is 0
        s.bitrate_kbps = 5000;
        assert!(s.is_live());
    }

    #[test]
    fn test_manager_add_and_count() {
        let mut mgr = IngestManager::new();
        mgr.add_source(make_source(1, IngestProtocol::Sdi));
        mgr.add_source(make_source(2, IngestProtocol::Ndi));
        assert_eq!(mgr.total_sources(), 2);
    }

    #[test]
    fn test_activate_source() {
        let mut mgr = IngestManager::new();
        mgr.add_source(make_source(1, IngestProtocol::Srt));
        mgr.activate(1).expect("should succeed in test");
        assert_eq!(mgr.active_sources().len(), 1);
    }

    #[test]
    fn test_activate_missing_source_err() {
        let mut mgr = IngestManager::new();
        assert!(mgr.activate(99).is_err());
    }

    #[test]
    fn test_activate_already_active_err() {
        let mut mgr = IngestManager::new();
        mgr.add_source(make_source(1, IngestProtocol::Rtmp));
        mgr.activate(1).expect("should succeed in test");
        assert!(mgr.activate(1).is_err());
    }

    #[test]
    fn test_deactivate_source() {
        let mut mgr = IngestManager::new();
        mgr.add_source(make_source(1, IngestProtocol::Rist));
        mgr.activate(1).expect("should succeed in test");
        mgr.deactivate(1).expect("should succeed in test");
        assert_eq!(mgr.active_sources().len(), 0);
    }

    #[test]
    fn test_deactivate_already_inactive_err() {
        let mut mgr = IngestManager::new();
        mgr.add_source(make_source(1, IngestProtocol::File));
        assert!(mgr.deactivate(1).is_err());
    }

    #[test]
    fn test_source_by_id_found() {
        let mut mgr = IngestManager::new();
        mgr.add_source(make_source(7, IngestProtocol::Ndi));
        assert!(mgr.source_by_id(7).is_some());
    }

    #[test]
    fn test_source_by_id_not_found() {
        let mgr = IngestManager::new();
        assert!(mgr.source_by_id(42).is_none());
    }

    #[test]
    fn test_ingest_stats_empty() {
        let mgr = IngestManager::new();
        let stats = ingest_stats(&mgr);
        assert_eq!(stats.total_sources, 0);
        assert_eq!(stats.active_sources, 0);
        assert_eq!(stats.total_bitrate_kbps, 0);
    }

    #[test]
    fn test_ingest_stats_with_active_source() {
        let mut mgr = IngestManager::new();
        let mut s = make_source(1, IngestProtocol::Rtmp);
        s.active = true;
        s.bitrate_kbps = 8000;
        mgr.add_source(s);
        let stats = ingest_stats(&mgr);
        assert_eq!(stats.active_sources, 1);
        assert_eq!(stats.total_bitrate_kbps, 8000);
    }

    #[test]
    fn test_protocol_display() {
        assert_eq!(IngestProtocol::Sdi.to_string(), "SDI");
        assert_eq!(IngestProtocol::Rtmp.to_string(), "RTMP");
        assert_eq!(IngestProtocol::File.to_string(), "File");
    }
}
