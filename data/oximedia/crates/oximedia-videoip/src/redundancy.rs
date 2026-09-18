//! IP video redundancy management (SMPTE 2022-7 / ST 2022-7).
//!
//! Provides hitless path switching, per-path health monitoring, and
//! packet deduplication for resilient professional video-over-IP transport.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// RedundancyMode
// ---------------------------------------------------------------------------

/// Redundancy mode for IP video streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedundancyMode {
    /// Single path — no redundancy.
    SinglePath,
    /// Dual path (SMPTE 2022-7 standard).
    DualPath,
    /// Triple path — maximum resilience.
    TriplePath,
}

impl RedundancyMode {
    /// Number of physical network paths used by this mode.
    #[must_use]
    pub fn path_count(&self) -> u8 {
        match self {
            Self::SinglePath => 1,
            Self::DualPath => 2,
            Self::TriplePath => 3,
        }
    }

    /// Maximum acceptable failover time in milliseconds for this mode.
    ///
    /// - `SinglePath`: no failover possible, returns `u32::MAX`.
    /// - `DualPath`: hitless SMPTE 2022-7 target of 0 ms (seamless merge).
    /// - `TriplePath`: 0 ms (seamless merge with extra path).
    #[must_use]
    pub fn failover_time_ms(&self) -> u32 {
        match self {
            Self::SinglePath => u32::MAX,
            Self::DualPath => 0,
            Self::TriplePath => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// PathStatus
// ---------------------------------------------------------------------------

/// Runtime health status for a single network path.
#[derive(Debug, Clone)]
pub struct PathStatus {
    /// Path identifier (0-based index).
    pub path_id: u8,
    /// Whether the path is currently active/enabled.
    pub active: bool,
    /// Observed packet loss percentage (0.0–100.0).
    pub packet_loss_pct: f32,
    /// Measured one-way latency in milliseconds.
    pub latency_ms: u32,
}

impl PathStatus {
    /// Returns `true` when the path is active **and** packet loss is below 1 %.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.active && self.packet_loss_pct < 1.0
    }
}

// ---------------------------------------------------------------------------
// RedundancyConfig
// ---------------------------------------------------------------------------

/// Configuration for a redundant IP video stream.
#[derive(Debug, Clone)]
pub struct RedundancyConfig {
    /// Redundancy mode to use.
    pub mode: RedundancyMode,
    /// Failover trigger threshold in milliseconds.
    pub switch_threshold_ms: u32,
    /// Whether hitless (seamless) switching is required.
    pub hitless: bool,
}

impl RedundancyConfig {
    /// Build a SMPTE 2022-7 compliant dual-path hitless configuration.
    #[must_use]
    pub fn smpte_2022_7() -> Self {
        Self {
            mode: RedundancyMode::DualPath,
            switch_threshold_ms: 0,
            hitless: true,
        }
    }
}

// ---------------------------------------------------------------------------
// PathMonitor
// ---------------------------------------------------------------------------

/// Monitors the health of one or more redundant network paths.
#[derive(Debug, Default)]
pub struct PathMonitor {
    /// All tracked paths.
    pub paths: Vec<PathStatus>,
}

impl PathMonitor {
    /// Create a new, empty path monitor.
    #[must_use]
    pub fn new() -> Self {
        Self { paths: Vec::new() }
    }

    /// Insert or update the status of `path_id` with fresh measurements.
    pub fn update_path(&mut self, id: u8, loss: f32, latency_ms: u32) {
        if let Some(p) = self.paths.iter_mut().find(|p| p.path_id == id) {
            p.packet_loss_pct = loss;
            p.latency_ms = latency_ms;
            p.active = true;
        } else {
            self.paths.push(PathStatus {
                path_id: id,
                active: true,
                packet_loss_pct: loss,
                latency_ms,
            });
        }
    }

    /// Returns references to all currently active paths.
    #[must_use]
    pub fn active_paths(&self) -> Vec<&PathStatus> {
        self.paths.iter().filter(|p| p.active).collect()
    }

    /// Returns the healthiest path with the lowest latency, if any.
    ///
    /// A candidate must be both active and have less than 1 % packet loss.
    #[must_use]
    pub fn best_path(&self) -> Option<&PathStatus> {
        self.paths
            .iter()
            .filter(|p| p.is_healthy())
            .min_by_key(|p| p.latency_ms)
    }

    /// Returns `true` when no path is healthy, indicating that manual
    /// intervention or failover to a backup system is required.
    #[must_use]
    pub fn should_failover(&self) -> bool {
        self.best_path().is_none()
    }
}

// ---------------------------------------------------------------------------
// PacketMerger
// ---------------------------------------------------------------------------

/// Merges duplicate RTP packets arriving via multiple redundant paths.
///
/// Each entry in `buffer` is `(sequence_number, payload)`.
/// Packets with the same sequence number are deduplicated; only the first
/// copy is retained for each sequence number.
#[derive(Debug)]
pub struct PacketMerger {
    /// Number of redundant paths feeding this merger.
    pub path_count: u8,
    /// Packet buffer: `(seq, data)` in insertion order.
    pub buffer: Vec<(u64, Vec<u8>)>,
}

impl PacketMerger {
    /// Create a new packet merger for `path_count` paths.
    #[must_use]
    pub fn new(path_count: u8) -> Self {
        Self {
            path_count,
            buffer: Vec::new(),
        }
    }

    /// Add a packet from `path_id` with sequence number `seq` and payload `data`.
    ///
    /// Duplicate sequence numbers (same packet from another path) are silently
    /// discarded.
    pub fn add_packet(&mut self, _path_id: u8, seq: u64, data: Vec<u8>) {
        if self.buffer.iter().any(|(s, _)| *s == seq) {
            return; // duplicate — discard
        }
        self.buffer.push((seq, data));
    }

    /// Remove and return the payload of the lowest-sequence packet in the
    /// buffer, or `None` if the buffer is empty.
    pub fn dequeue(&mut self) -> Option<Vec<u8>> {
        if self.buffer.is_empty() {
            return None;
        }
        // Find index of the entry with the smallest sequence number.
        let min_idx = self
            .buffer
            .iter()
            .enumerate()
            .min_by_key(|(_, (seq, _))| *seq)
            .map(|(i, _)| i)?;
        Some(self.buffer.remove(min_idx).1)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // RedundancyMode tests

    #[test]
    fn test_single_path_count() {
        assert_eq!(RedundancyMode::SinglePath.path_count(), 1);
    }

    #[test]
    fn test_dual_path_count() {
        assert_eq!(RedundancyMode::DualPath.path_count(), 2);
    }

    #[test]
    fn test_triple_path_count() {
        assert_eq!(RedundancyMode::TriplePath.path_count(), 3);
    }

    #[test]
    fn test_single_path_failover_time_max() {
        assert_eq!(RedundancyMode::SinglePath.failover_time_ms(), u32::MAX);
    }

    #[test]
    fn test_dual_path_failover_time_zero() {
        assert_eq!(RedundancyMode::DualPath.failover_time_ms(), 0);
    }

    #[test]
    fn test_triple_path_failover_time_zero() {
        assert_eq!(RedundancyMode::TriplePath.failover_time_ms(), 0);
    }

    // PathStatus tests

    #[test]
    fn test_path_status_healthy() {
        let p = PathStatus {
            path_id: 0,
            active: true,
            packet_loss_pct: 0.5,
            latency_ms: 10,
        };
        assert!(p.is_healthy());
    }

    #[test]
    fn test_path_status_unhealthy_inactive() {
        let p = PathStatus {
            path_id: 0,
            active: false,
            packet_loss_pct: 0.0,
            latency_ms: 5,
        };
        assert!(!p.is_healthy());
    }

    #[test]
    fn test_path_status_unhealthy_high_loss() {
        let p = PathStatus {
            path_id: 1,
            active: true,
            packet_loss_pct: 2.0,
            latency_ms: 5,
        };
        assert!(!p.is_healthy());
    }

    // RedundancyConfig tests

    #[test]
    fn test_smpte_2022_7_config() {
        let cfg = RedundancyConfig::smpte_2022_7();
        assert_eq!(cfg.mode, RedundancyMode::DualPath);
        assert!(cfg.hitless);
        assert_eq!(cfg.switch_threshold_ms, 0);
    }

    // PathMonitor tests

    #[test]
    fn test_path_monitor_update_creates_path() {
        let mut m = PathMonitor::new();
        m.update_path(0, 0.0, 10);
        assert_eq!(m.paths.len(), 1);
    }

    #[test]
    fn test_path_monitor_update_modifies_existing() {
        let mut m = PathMonitor::new();
        m.update_path(0, 0.5, 10);
        m.update_path(0, 2.0, 20);
        assert_eq!(m.paths.len(), 1);
        assert!((m.paths[0].packet_loss_pct - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_path_monitor_active_paths() {
        let mut m = PathMonitor::new();
        m.update_path(0, 0.0, 5);
        m.update_path(1, 0.0, 8);
        m.paths[1].active = false;
        assert_eq!(m.active_paths().len(), 1);
    }

    #[test]
    fn test_path_monitor_best_path_min_latency() {
        let mut m = PathMonitor::new();
        m.update_path(0, 0.0, 20);
        m.update_path(1, 0.0, 10);
        let best = m.best_path().expect("should succeed in test");
        assert_eq!(best.path_id, 1);
    }

    #[test]
    fn test_path_monitor_best_path_none_unhealthy() {
        let mut m = PathMonitor::new();
        m.update_path(0, 5.0, 10); // loss too high
        assert!(m.best_path().is_none());
    }

    #[test]
    fn test_path_monitor_should_failover_true() {
        let mut m = PathMonitor::new();
        m.update_path(0, 99.0, 10);
        assert!(m.should_failover());
    }

    #[test]
    fn test_path_monitor_should_failover_false() {
        let mut m = PathMonitor::new();
        m.update_path(0, 0.0, 10);
        assert!(!m.should_failover());
    }

    // PacketMerger tests

    #[test]
    fn test_packet_merger_dedup() {
        let mut pm = PacketMerger::new(2);
        pm.add_packet(0, 1, vec![1, 2, 3]);
        pm.add_packet(1, 1, vec![1, 2, 3]); // duplicate
        assert_eq!(pm.buffer.len(), 1);
    }

    #[test]
    fn test_packet_merger_dequeue_order() {
        let mut pm = PacketMerger::new(2);
        pm.add_packet(0, 5, vec![5]);
        pm.add_packet(1, 3, vec![3]);
        pm.add_packet(0, 4, vec![4]);
        assert_eq!(pm.dequeue(), Some(vec![3]));
        assert_eq!(pm.dequeue(), Some(vec![4]));
        assert_eq!(pm.dequeue(), Some(vec![5]));
        assert_eq!(pm.dequeue(), None);
    }

    #[test]
    fn test_packet_merger_empty_dequeue() {
        let mut pm = PacketMerger::new(1);
        assert_eq!(pm.dequeue(), None);
    }
}
