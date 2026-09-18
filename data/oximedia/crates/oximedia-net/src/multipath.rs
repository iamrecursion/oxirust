//! Multipath streaming — redundant transmission over multiple network interfaces.
//!
//! Multipath streaming sends the same (or split) media stream over several
//! independent network paths simultaneously.  The receiver selects the best
//! arriving copy of each packet and discards duplicates.
//!
//! Use cases:
//! - **Broadcast reliability**: simultaneously transmit over LTE + Ethernet + WiFi.
//! - **Bonding**: aggregate multiple low-bandwidth links for higher throughput.
//! - **Hot-standby**: keep a secondary path warm so failover is instant.
//!
//! This module provides:
//! - [`PathHandle`] — an abstraction for a network path (interface + address).
//! - [`MultipathSender`] — schedules packets across paths and detects congestion.
//! - [`MultipathReceiver`] — de-duplicates packets from multiple paths.
//! - Scheduling strategies: Round-Robin, Redundant, Bandwidth-Weighted.
//! - Per-path health monitoring (RTT, loss, throughput).

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

// ─── Scheduling Strategy ──────────────────────────────────────────────────────

/// How packets are distributed across available paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingStrategy {
    /// Send every packet on all available paths (maximum redundancy).
    Redundant,
    /// Distribute packets evenly in a round-robin fashion.
    RoundRobin,
    /// Distribute packets proportionally to each path's estimated bandwidth.
    BandwidthWeighted,
    /// Send on the single highest-quality path; failover on loss.
    ActiveStandby,
}

impl SchedulingStrategy {
    /// Returns a human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Redundant => "redundant",
            Self::RoundRobin => "round-robin",
            Self::BandwidthWeighted => "bandwidth-weighted",
            Self::ActiveStandby => "active-standby",
        }
    }
}

// ─── Path Status ─────────────────────────────────────────────────────────────

/// Operational status of a network path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathStatus {
    /// Path is active and usable.
    Up,
    /// Path is experiencing degraded performance.
    Degraded,
    /// Path is down (no connectivity).
    Down,
    /// Path is in standby (not actively used).
    Standby,
}

// ─── Path Health ──────────────────────────────────────────────────────────────

/// Health metrics for a single network path.
#[derive(Debug, Clone)]
pub struct PathHealth {
    /// Smoothed round-trip time (SRTT).
    pub rtt: Duration,
    /// RTT variance.
    pub rtt_variance: Duration,
    /// Estimated one-way delay.
    pub one_way_delay: Duration,
    /// Packet loss rate (0.0 – 1.0).
    pub loss_rate: f64,
    /// Estimated available bandwidth in bits per second.
    pub bandwidth_bps: f64,
    /// Jitter in microseconds.
    pub jitter_us: f64,
    /// Path status.
    pub status: PathStatus,
    /// Last update time.
    pub last_updated: Instant,
}

impl Default for PathHealth {
    fn default() -> Self {
        Self {
            rtt: Duration::from_millis(10),
            rtt_variance: Duration::from_millis(1),
            one_way_delay: Duration::from_millis(5),
            loss_rate: 0.0,
            bandwidth_bps: 10_000_000.0, // 10 Mbps default
            jitter_us: 0.0,
            status: PathStatus::Up,
            last_updated: Instant::now(),
        }
    }
}

impl PathHealth {
    /// Computes a composite quality score [0.0, 1.0].
    ///
    /// Higher is better.
    #[must_use]
    pub fn quality_score(&self) -> f64 {
        let rtt_penalty = (self.rtt.as_millis() as f64 / 200.0).clamp(0.0, 1.0);
        let loss_penalty = self.loss_rate.clamp(0.0, 1.0);
        let jitter_penalty = (self.jitter_us / 10_000.0).clamp(0.0, 1.0);

        let raw = 1.0 - (rtt_penalty * 0.4 + loss_penalty * 0.4 + jitter_penalty * 0.2);
        raw.clamp(0.0, 1.0)
    }

    /// Updates the RTT using an EWMA with α = 1/8 (RFC 6298).
    pub fn update_rtt(&mut self, sample: Duration) {
        let alpha_8 = sample / 8;
        self.rtt = self.rtt - (self.rtt / 8) + alpha_8;
        let abs_diff = if sample > self.rtt {
            sample - self.rtt
        } else {
            self.rtt - sample
        };
        self.rtt_variance = self.rtt_variance - (self.rtt_variance / 4) + abs_diff / 4;
        self.last_updated = Instant::now();
    }

    /// Updates the loss rate using an EWMA.
    pub fn update_loss(&mut self, lost: bool) {
        let sample = if lost { 1.0 } else { 0.0 };
        self.loss_rate = 0.875 * self.loss_rate + 0.125 * sample;
        if self.loss_rate > 0.5 {
            self.status = PathStatus::Degraded;
        } else {
            self.status = PathStatus::Up;
        }
        self.last_updated = Instant::now();
    }
}

// ─── Path Handle ─────────────────────────────────────────────────────────────

/// Represents a single network path (interface + peer address).
#[derive(Debug, Clone)]
pub struct PathHandle {
    /// Unique path identifier.
    pub id: u32,
    /// Local bind address (interface).
    pub local_addr: SocketAddr,
    /// Remote peer address.
    pub remote_addr: SocketAddr,
    /// Human-readable label (e.g., "eth0", "wlan0", "lte0").
    pub label: String,
    /// Whether this is the primary path.
    pub is_primary: bool,
    /// Current health metrics.
    pub health: PathHealth,
    /// Total bytes sent on this path.
    pub bytes_sent: u64,
    /// Total packets sent on this path.
    pub packets_sent: u64,
}

impl PathHandle {
    /// Creates a new path handle.
    #[must_use]
    pub fn new(
        id: u32,
        local_addr: SocketAddr,
        remote_addr: SocketAddr,
        label: impl Into<String>,
    ) -> Self {
        Self {
            id,
            local_addr,
            remote_addr,
            label: label.into(),
            is_primary: id == 0,
            health: PathHealth::default(),
            bytes_sent: 0,
            packets_sent: 0,
        }
    }

    /// Returns whether this path is usable.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        matches!(self.health.status, PathStatus::Up | PathStatus::Degraded)
    }
}

// ─── Multipath Configuration ──────────────────────────────────────────────────

/// Configuration for the multipath sender/receiver.
#[derive(Debug, Clone)]
pub struct MultipathConfig {
    /// Packet scheduling strategy.
    pub strategy: SchedulingStrategy,
    /// Maximum de-duplication window (number of sequence numbers).
    pub dedup_window: u32,
    /// Merge delay for out-of-order packets.
    pub merge_delay: Duration,
    /// Path health probe interval.
    pub probe_interval: Duration,
    /// Threshold loss rate to mark a path as degraded.
    pub degraded_loss_threshold: f64,
    /// Threshold loss rate to mark a path as down.
    pub down_loss_threshold: f64,
}

impl Default for MultipathConfig {
    fn default() -> Self {
        Self {
            strategy: SchedulingStrategy::Redundant,
            dedup_window: 512,
            merge_delay: Duration::from_millis(30),
            probe_interval: Duration::from_millis(200),
            degraded_loss_threshold: 0.05,
            down_loss_threshold: 0.30,
        }
    }
}

impl MultipathConfig {
    /// Creates a new configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the scheduling strategy.
    #[must_use]
    pub const fn with_strategy(mut self, s: SchedulingStrategy) -> Self {
        self.strategy = s;
        self
    }
}

// ─── Multipath Sender ─────────────────────────────────────────────────────────

/// Multipath sender: dispatches packets across multiple network paths.
pub struct MultipathSender {
    /// Configuration.
    config: MultipathConfig,
    /// Available paths.
    paths: Vec<PathHandle>,
    /// Round-robin index.
    rr_index: usize,
    /// Total packets dispatched.
    total_dispatched: u64,
    /// Next sequence number.
    next_seq: u32,
}

impl MultipathSender {
    /// Creates a new multipath sender.
    #[must_use]
    pub fn new(config: MultipathConfig) -> Self {
        Self {
            config,
            paths: Vec::new(),
            rr_index: 0,
            total_dispatched: 0,
            next_seq: 0,
        }
    }

    /// Adds a network path.
    pub fn add_path(&mut self, path: PathHandle) {
        self.paths.push(path);
    }

    /// Removes a path by ID.
    pub fn remove_path(&mut self, path_id: u32) {
        self.paths.retain(|p| p.id != path_id);
    }

    /// Returns the number of active paths.
    #[must_use]
    pub fn path_count(&self) -> usize {
        self.paths.len()
    }

    /// Returns the number of usable paths.
    #[must_use]
    pub fn usable_path_count(&self) -> usize {
        self.paths.iter().filter(|p| p.is_usable()).count()
    }

    /// Schedules a packet for transmission.
    ///
    /// Returns the list of `(path_id, destination_addr)` pairs where the
    /// packet should be sent.  The caller performs the actual socket writes.
    pub fn schedule(&mut self, payload_len: usize) -> Vec<(u32, SocketAddr)> {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        self.total_dispatched += 1;

        let usable: Vec<usize> = self
            .paths
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_usable())
            .map(|(i, _)| i)
            .collect();

        if usable.is_empty() {
            return Vec::new();
        }

        let mut out = Vec::new();

        match self.config.strategy {
            SchedulingStrategy::Redundant => {
                for &idx in &usable {
                    self.paths[idx].packets_sent += 1;
                    self.paths[idx].bytes_sent += payload_len as u64;
                    out.push((self.paths[idx].id, self.paths[idx].remote_addr));
                }
            }

            SchedulingStrategy::RoundRobin => {
                // Find next usable path in round-robin order.
                let start = self.rr_index % usable.len();
                let idx = usable[start];
                self.rr_index = (self.rr_index + 1) % usable.len();
                self.paths[idx].packets_sent += 1;
                self.paths[idx].bytes_sent += payload_len as u64;
                out.push((self.paths[idx].id, self.paths[idx].remote_addr));
            }

            SchedulingStrategy::BandwidthWeighted => {
                // Select the path with the highest estimated bandwidth.
                let best_idx = usable
                    .iter()
                    .copied()
                    .max_by(|&a, &b| {
                        self.paths[a]
                            .health
                            .bandwidth_bps
                            .partial_cmp(&self.paths[b].health.bandwidth_bps)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(usable[0]);
                self.paths[best_idx].packets_sent += 1;
                self.paths[best_idx].bytes_sent += payload_len as u64;
                out.push((self.paths[best_idx].id, self.paths[best_idx].remote_addr));
            }

            SchedulingStrategy::ActiveStandby => {
                // Use the primary path; fallback to the best standby if primary is down.
                let primary_idx = usable.iter().copied().find(|&i| self.paths[i].is_primary);
                let idx = primary_idx.unwrap_or(usable[0]);
                self.paths[idx].packets_sent += 1;
                self.paths[idx].bytes_sent += payload_len as u64;
                out.push((self.paths[idx].id, self.paths[idx].remote_addr));
            }
        }

        let _ = seq; // Used for future sequence tracking
        out
    }

    /// Updates path health after an RTT measurement.
    pub fn update_path_rtt(&mut self, path_id: u32, rtt: Duration) {
        if let Some(p) = self.paths.iter_mut().find(|p| p.id == path_id) {
            p.health.update_rtt(rtt);
        }
    }

    /// Reports a packet loss event on a path.
    pub fn report_path_loss(&mut self, path_id: u32, lost: bool) {
        if let Some(p) = self.paths.iter_mut().find(|p| p.id == path_id) {
            p.health.update_loss(lost);
        }
    }

    /// Returns a reference to all paths.
    #[must_use]
    pub fn paths(&self) -> &[PathHandle] {
        &self.paths
    }

    /// Returns the best path (highest quality score).
    #[must_use]
    pub fn best_path(&self) -> Option<&PathHandle> {
        self.paths.iter().filter(|p| p.is_usable()).max_by(|a, b| {
            a.health
                .quality_score()
                .partial_cmp(&b.health.quality_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns total dispatched packets.
    #[must_use]
    pub const fn total_dispatched(&self) -> u64 {
        self.total_dispatched
    }
}

// ─── Multipath Receiver ───────────────────────────────────────────────────────

/// Multipath receiver: de-duplicates packets arriving from multiple paths.
pub struct MultipathReceiver {
    /// Configuration.
    config: MultipathConfig,
    /// De-duplication ring buffer: seq → arrival time.
    dedup_seen: BTreeMap<u32, Instant>,
    /// Packets ready for delivery (seq → payload).
    ready: BTreeMap<u32, Vec<u8>>,
    /// Per-path statistics (path_id → (received, duplicates)).
    path_stats: HashMap<u32, (u64, u64)>,
    /// Total delivered.
    total_delivered: u64,
    /// Highest sequence number seen.
    highest_seq: Option<u32>,
}

impl MultipathReceiver {
    /// Creates a new multipath receiver.
    #[must_use]
    pub fn new(config: MultipathConfig) -> Self {
        Self {
            config,
            dedup_seen: BTreeMap::new(),
            ready: BTreeMap::new(),
            path_stats: HashMap::new(),
            total_delivered: 0,
            highest_seq: None,
        }
    }

    /// Processes an incoming packet from the given path.
    ///
    /// Returns `true` if this is the first copy of the packet.
    pub fn receive(&mut self, seq: u32, payload: Vec<u8>, path_id: u32) -> bool {
        let stats = self.path_stats.entry(path_id).or_insert((0, 0));
        stats.0 += 1;

        if self.dedup_seen.contains_key(&seq) {
            stats.1 += 1;
            return false;
        }

        // Evict old entries beyond the window.
        let window = self.config.dedup_window;
        if self.dedup_seen.len() >= window as usize {
            if let Some((&oldest_seq, _)) = self.dedup_seen.iter().next() {
                self.dedup_seen.remove(&oldest_seq);
            }
        }

        self.dedup_seen.insert(seq, Instant::now());
        self.ready.insert(seq, payload);

        match self.highest_seq {
            None => self.highest_seq = Some(seq),
            Some(h) if seq.wrapping_sub(h) < 32768 => self.highest_seq = Some(seq),
            _ => {}
        }

        true
    }

    /// Drains packets that have waited at least the merge delay.
    pub fn drain_ready(&mut self) -> Vec<(u32, Vec<u8>)> {
        let delay = self.config.merge_delay;
        let mut out = Vec::new();

        let ready_seqs: Vec<u32> = self
            .dedup_seen
            .iter()
            .filter(|(_, t)| t.elapsed() >= delay)
            .filter(|(seq, _)| self.ready.contains_key(seq))
            .map(|(&seq, _)| seq)
            .collect();

        for seq in ready_seqs {
            if let Some(payload) = self.ready.remove(&seq) {
                out.push((seq, payload));
                self.total_delivered += 1;
            }
        }

        out
    }

    /// Returns per-path statistics `(received, duplicates)`.
    #[must_use]
    pub fn path_stats(&self, path_id: u32) -> Option<(u64, u64)> {
        self.path_stats.get(&path_id).copied()
    }

    /// Returns total delivered packets.
    #[must_use]
    pub const fn total_delivered(&self) -> u64 {
        self.total_delivered
    }

    /// Returns the number of packets pending delivery.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.ready.len()
    }
}

// ─── High-level Multipath Streaming API ──────────────────────────────────────
//
// The types below implement the simplified multipath API described in the
// module documentation. They complement the lower-level `MultipathSender` /
// `MultipathReceiver` pair above with an endpoint-list configuration model
// that mirrors the public task specification.

/// A single network endpoint in a multipath configuration.
#[derive(Debug, Clone)]
pub struct MultipathEndpoint {
    /// Target address in `host:port` format, e.g. `"192.0.2.1:5004"`.
    pub address: String,
    /// Relative scheduling weight in the range `0.0 – 1.0`.
    /// Used by [`MultipathScheduler::WeightedRoundRobin`].
    pub weight: f32,
    /// Optional NIC name to bind the socket to (e.g. `"eth0"`, `"wlan0"`).
    pub interface: Option<String>,
}

impl MultipathEndpoint {
    /// Creates a new endpoint with the given address and weight.
    #[must_use]
    pub fn new(address: impl Into<String>, weight: f32) -> Self {
        Self {
            address: address.into(),
            weight: weight.clamp(0.0, 1.0),
            interface: None,
        }
    }

    /// Attaches a NIC binding to this endpoint.
    #[must_use]
    pub fn with_interface(mut self, iface: impl Into<String>) -> Self {
        self.interface = Some(iface.into());
        self
    }
}

// ─── MultipathScheduler ───────────────────────────────────────────────────────

/// Packet scheduling strategy for the multipath stream sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultipathScheduler {
    /// Distribute packets in strict rotation across all active endpoints.
    RoundRobin,
    /// Distribute packets proportionally to each endpoint's [`MultipathEndpoint::weight`].
    WeightedRoundRobin,
    /// Always send to the endpoint that last reported the lowest RTT.
    MinLatency,
    /// Send every packet on all active endpoints simultaneously (maximum redundancy).
    Redundant,
}

// ─── RedundancyMode ───────────────────────────────────────────────────────────

/// Controls how data is protected across multiple paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedundancyMode {
    /// No redundancy — only the path chosen by the scheduler receives data.
    None,
    /// Forward Error Correction across paths (XOR-based, RFC 5109 style).
    Fec,
    /// Duplicate every packet on all paths.
    Duplicate,
}

// ─── Per-path Statistics ──────────────────────────────────────────────────────

/// Live statistics for a single network path in the stream sender.
#[derive(Debug, Clone, Default)]
pub struct PathStats {
    /// Total bytes sent on this path.
    pub bytes_sent: u64,
    /// Total bytes received (acknowledgement-based) on this path.
    pub bytes_received: u64,
    /// Latest round-trip time measurement in milliseconds.
    pub rtt_ms: f32,
    /// Packet loss rate in the range `0.0 – 1.0` (EWMA-smoothed).
    pub packet_loss_rate: f32,
    /// Whether the path is currently considered active.
    pub active: bool,
}

impl PathStats {
    /// Creates a new active `PathStats` with zero counters.
    #[must_use]
    pub fn new_active() -> Self {
        Self {
            active: true,
            ..Default::default()
        }
    }

    /// Updates `rtt_ms` using an exponential moving average (α = 0.125).
    pub fn update_rtt(&mut self, sample_ms: f32) {
        if self.rtt_ms == 0.0 {
            self.rtt_ms = sample_ms;
        } else {
            self.rtt_ms = self.rtt_ms * 0.875 + sample_ms * 0.125;
        }
    }

    /// Updates `packet_loss_rate` using an EWMA (α = 0.125).
    pub fn update_loss(&mut self, loss_rate: f32) {
        self.packet_loss_rate = self.packet_loss_rate * 0.875 + loss_rate.clamp(0.0, 1.0) * 0.125;
    }
}

// ─── Aggregate Statistics ─────────────────────────────────────────────────────

/// Aggregate statistics for a [`MultipathStreamSender`].
#[derive(Debug, Clone, Default)]
pub struct MultipathStats {
    /// Per-path statistics, indexed parallel to the endpoint list.
    pub path_stats: Vec<PathStats>,
    /// Total bytes sent across all paths.
    pub total_bytes_sent: u64,
    /// Total bytes received (ACKed) across all paths.
    pub total_bytes_received: u64,
    /// Number of packets that arrived out-of-order at the receiver.
    pub reordered_packets: u64,
    /// Number of packets recovered via FEC or path redundancy.
    pub recovered_packets: u64,
}

impl MultipathStats {
    /// Creates a `MultipathStats` with one `PathStats` per endpoint.
    #[must_use]
    pub fn for_endpoints(count: usize) -> Self {
        Self {
            path_stats: (0..count).map(|_| PathStats::new_active()).collect(),
            ..Default::default()
        }
    }
}

// ─── High-level Config ────────────────────────────────────────────────────────

/// High-level configuration for the stream-oriented multipath sender.
///
/// Pairs a list of [`MultipathEndpoint`]s with a scheduling and redundancy
/// policy, mirroring the public task API.
#[derive(Debug, Clone)]
pub struct MultipathStreamConfig {
    /// Ordered list of network endpoints.
    pub paths: Vec<MultipathEndpoint>,
    /// Packet-scheduling strategy.
    pub scheduler: MultipathScheduler,
    /// Redundancy mode controlling how data is protected.
    pub redundancy_mode: RedundancyMode,
}

impl MultipathStreamConfig {
    /// Creates a new configuration.
    #[must_use]
    pub fn new(
        paths: Vec<MultipathEndpoint>,
        scheduler: MultipathScheduler,
        redundancy_mode: RedundancyMode,
    ) -> Self {
        Self {
            paths,
            scheduler,
            redundancy_mode,
        }
    }
}

// ─── MultipathStreamSender ────────────────────────────────────────────────────

/// Stream-oriented multipath sender.
///
/// Manages a list of [`MultipathEndpoint`]s, selects paths according to the
/// configured [`MultipathScheduler`], and maintains live [`MultipathStats`].
///
/// This is the high-level counterpart to the lower-level [`MultipathSender`]
/// which operates on [`PathHandle`] objects.
pub struct MultipathStreamSender {
    config: MultipathStreamConfig,
    stats: MultipathStats,
    round_robin_idx: usize,
    /// Weighted round-robin: accumulated weights for fair scheduling.
    wrr_counters: Vec<f32>,
}

impl MultipathStreamSender {
    /// Creates a new sender from the given configuration.
    ///
    /// All paths start as active.
    #[must_use]
    pub fn new(config: MultipathStreamConfig) -> Self {
        let path_count = config.paths.len();
        let wrr_counters = config.paths.iter().map(|_| 0.0f32).collect();
        Self {
            stats: MultipathStats::for_endpoints(path_count),
            config,
            round_robin_idx: 0,
            wrr_counters,
        }
    }

    /// Selects the index of the next path to use for transmission.
    ///
    /// The selection algorithm is determined by
    /// [`MultipathStreamConfig::scheduler`].
    ///
    /// Returns `0` if no active paths are available (safe fallback).
    pub fn select_path(&mut self) -> usize {
        let active: Vec<usize> = self
            .stats
            .path_stats
            .iter()
            .enumerate()
            .filter(|(_, s)| s.active)
            .map(|(i, _)| i)
            .collect();

        if active.is_empty() {
            return 0;
        }

        match self.config.scheduler {
            MultipathScheduler::RoundRobin => {
                let pos = self.round_robin_idx % active.len();
                let idx = active[pos];
                self.round_robin_idx = self.round_robin_idx.wrapping_add(1);
                idx
            }

            MultipathScheduler::WeightedRoundRobin => {
                // Increment each counter by its weight; pick highest.
                let total_weight: f32 = active
                    .iter()
                    .map(|&i| self.config.paths[i].weight)
                    .sum::<f32>()
                    .max(f32::EPSILON);

                for &i in &active {
                    self.wrr_counters[i] += self.config.paths[i].weight / total_weight;
                }

                let best = active
                    .iter()
                    .copied()
                    .max_by(|&a, &b| {
                        self.wrr_counters[a]
                            .partial_cmp(&self.wrr_counters[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(active[0]);

                self.wrr_counters[best] -= 1.0;
                best
            }

            MultipathScheduler::MinLatency => {
                // Pick the active path with the lowest RTT.
                // Treat rtt_ms == 0.0 as "not yet measured" and fall back to index.
                active
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        let ra = if self.stats.path_stats[a].rtt_ms > 0.0 {
                            self.stats.path_stats[a].rtt_ms
                        } else {
                            f32::MAX
                        };
                        let rb = if self.stats.path_stats[b].rtt_ms > 0.0 {
                            self.stats.path_stats[b].rtt_ms
                        } else {
                            f32::MAX
                        };
                        ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(active[0])
            }

            MultipathScheduler::Redundant => {
                // Redundant mode — caller is expected to send on all paths.
                // We return index 0 here; use `active_path_indices()` for all.
                active[0]
            }
        }
    }

    /// Returns the indices of all currently active paths.
    ///
    /// Useful when [`MultipathScheduler::Redundant`] is active and the caller
    /// needs to send the same data on every path.
    #[must_use]
    pub fn active_path_indices(&self) -> Vec<usize> {
        self.stats
            .path_stats
            .iter()
            .enumerate()
            .filter(|(_, s)| s.active)
            .map(|(i, _)| i)
            .collect()
    }

    /// Updates the RTT measurement for the given path.
    ///
    /// Silently ignores out-of-range indices.
    pub fn update_path_rtt(&mut self, path_idx: usize, rtt_ms: f32) {
        if let Some(ps) = self.stats.path_stats.get_mut(path_idx) {
            ps.update_rtt(rtt_ms);
        }
    }

    /// Updates the packet-loss rate for the given path.
    ///
    /// Silently ignores out-of-range indices.
    pub fn update_path_loss(&mut self, path_idx: usize, loss_rate: f32) {
        if let Some(ps) = self.stats.path_stats.get_mut(path_idx) {
            ps.update_loss(loss_rate);
            // Mark as inactive if loss is catastrophic (> 90 %).
            if ps.packet_loss_rate > 0.90 {
                ps.active = false;
            }
        }
    }

    /// Records that `bytes` were transmitted on the given path.
    pub fn record_bytes_sent(&mut self, path_idx: usize, bytes: u64) {
        if let Some(ps) = self.stats.path_stats.get_mut(path_idx) {
            ps.bytes_sent = ps.bytes_sent.saturating_add(bytes);
            self.stats.total_bytes_sent = self.stats.total_bytes_sent.saturating_add(bytes);
        }
    }

    /// Returns a reference to the live aggregate statistics.
    #[must_use]
    pub fn stats(&self) -> &MultipathStats {
        &self.stats
    }

    /// Returns the number of active paths.
    #[must_use]
    pub fn active_path_count(&self) -> usize {
        self.stats.path_stats.iter().filter(|s| s.active).count()
    }

    /// Marks a path as active or inactive by index.
    pub fn set_path_active(&mut self, path_idx: usize, active: bool) {
        if let Some(ps) = self.stats.path_stats.get_mut(path_idx) {
            ps.active = active;
        }
    }

    /// Returns a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &MultipathStreamConfig {
        &self.config
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_path(id: u32) -> PathHandle {
        PathHandle::new(
            id,
            "127.0.0.1:5000".parse().expect("valid addr"),
            "192.168.1.1:5000".parse().expect("valid addr"),
            format!("eth{id}"),
        )
    }

    fn default_config() -> MultipathConfig {
        MultipathConfig::default()
    }

    // 1. Strategy names
    #[test]
    fn test_strategy_names() {
        assert_eq!(SchedulingStrategy::Redundant.name(), "redundant");
        assert_eq!(SchedulingStrategy::RoundRobin.name(), "round-robin");
        assert_eq!(
            SchedulingStrategy::BandwidthWeighted.name(),
            "bandwidth-weighted"
        );
        assert_eq!(SchedulingStrategy::ActiveStandby.name(), "active-standby");
    }

    // 2. PathHealth default quality score
    #[test]
    fn test_path_health_quality_score_default() {
        let h = PathHealth::default();
        let score = h.quality_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    // 3. PathHealth quality decreases with loss
    #[test]
    fn test_path_health_quality_decreases_with_loss() {
        let h_good = PathHealth::default();
        let mut h_bad = PathHealth::default();
        h_bad.loss_rate = 0.5;
        assert!(h_good.quality_score() > h_bad.quality_score());
    }

    // 4. PathHealth RTT update
    #[test]
    fn test_path_health_rtt_update() {
        let mut h = PathHealth::default();
        let old_rtt = h.rtt;
        h.update_rtt(Duration::from_millis(100));
        // RTT should be a weighted average
        assert_ne!(h.rtt, old_rtt);
    }

    // 5. PathHealth loss rate update
    #[test]
    fn test_path_health_loss_update() {
        let mut h = PathHealth::default();
        assert_eq!(h.loss_rate, 0.0);
        h.update_loss(true);
        assert!(h.loss_rate > 0.0);
    }

    // 6. PathHandle is_usable
    #[test]
    fn test_path_handle_is_usable() {
        let p = make_path(0);
        assert!(p.is_usable());
    }

    // 7. MultipathSender add path
    #[test]
    fn test_sender_add_path() {
        let mut sender = MultipathSender::new(default_config());
        sender.add_path(make_path(0));
        sender.add_path(make_path(1));
        assert_eq!(sender.path_count(), 2);
    }

    // 8. Redundant strategy sends on all paths
    #[test]
    fn test_sender_redundant_strategy() {
        let cfg = MultipathConfig::new().with_strategy(SchedulingStrategy::Redundant);
        let mut sender = MultipathSender::new(cfg);
        sender.add_path(make_path(0));
        sender.add_path(make_path(1));
        let dispatched = sender.schedule(188);
        assert_eq!(dispatched.len(), 2);
    }

    // 9. RoundRobin cycles through paths
    #[test]
    fn test_sender_round_robin() {
        let cfg = MultipathConfig::new().with_strategy(SchedulingStrategy::RoundRobin);
        let mut sender = MultipathSender::new(cfg);
        sender.add_path(make_path(0));
        sender.add_path(make_path(1));
        let d1 = sender.schedule(188);
        let d2 = sender.schedule(188);
        assert_eq!(d1.len(), 1);
        assert_eq!(d2.len(), 1);
        // Should alternate paths
        assert_ne!(d1[0].0, d2[0].0);
    }

    // 10. BandwidthWeighted selects highest bandwidth path
    #[test]
    fn test_sender_bandwidth_weighted() {
        let cfg = MultipathConfig::new().with_strategy(SchedulingStrategy::BandwidthWeighted);
        let mut sender = MultipathSender::new(cfg);
        let mut p0 = make_path(0);
        p0.health.bandwidth_bps = 1_000_000.0;
        let mut p1 = make_path(1);
        p1.health.bandwidth_bps = 10_000_000.0;
        sender.add_path(p0);
        sender.add_path(p1);
        let dispatched = sender.schedule(188);
        assert_eq!(dispatched[0].0, 1); // Path 1 has higher bandwidth
    }

    // 11. ActiveStandby prefers primary path
    #[test]
    fn test_sender_active_standby() {
        let cfg = MultipathConfig::new().with_strategy(SchedulingStrategy::ActiveStandby);
        let mut sender = MultipathSender::new(cfg);
        let p0 = make_path(0); // id=0 → is_primary=true
        let p1 = make_path(1);
        sender.add_path(p0);
        sender.add_path(p1);
        let dispatched = sender.schedule(188);
        assert_eq!(dispatched[0].0, 0); // Primary path
    }

    // 12. Remove path
    #[test]
    fn test_sender_remove_path() {
        let mut sender = MultipathSender::new(default_config());
        sender.add_path(make_path(0));
        sender.add_path(make_path(1));
        sender.remove_path(1);
        assert_eq!(sender.path_count(), 1);
    }

    // 13. No paths → empty schedule
    #[test]
    fn test_sender_no_paths() {
        let mut sender = MultipathSender::new(default_config());
        let dispatched = sender.schedule(188);
        assert!(dispatched.is_empty());
    }

    // 14. MultipathReceiver de-duplication
    #[test]
    fn test_receiver_dedup() {
        let mut rx = MultipathReceiver::new(default_config());
        assert!(rx.receive(0, vec![0u8], 0)); // First copy
        assert!(!rx.receive(0, vec![0u8], 1)); // Duplicate
    }

    // 15. MultipathReceiver drain after merge delay
    #[test]
    fn test_receiver_drain_after_delay() {
        let cfg = MultipathConfig {
            merge_delay: Duration::from_millis(1),
            ..Default::default()
        };
        let mut rx = MultipathReceiver::new(cfg);
        rx.receive(0, vec![42u8], 0);
        std::thread::sleep(Duration::from_millis(5));
        let drained = rx.drain_ready();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, 0);
    }

    // 16. Per-path statistics
    #[test]
    fn test_receiver_path_stats() {
        let mut rx = MultipathReceiver::new(default_config());
        rx.receive(0, vec![0u8], 0);
        rx.receive(0, vec![0u8], 1); // Duplicate on path 1
        let (recv, dups) = rx.path_stats(1).expect("path 1 should have stats");
        assert_eq!(recv, 1);
        assert_eq!(dups, 1);
    }

    // 17. Best path selection
    #[test]
    fn test_sender_best_path() {
        let mut sender = MultipathSender::new(default_config());
        let mut p0 = make_path(0);
        p0.health.loss_rate = 0.5; // Degraded
        let p1 = make_path(1); // Healthy
        sender.add_path(p0);
        sender.add_path(p1);
        let best = sender.best_path().expect("should have a best path");
        assert_eq!(best.id, 1);
    }

    // 18. Update path RTT
    #[test]
    fn test_sender_update_path_rtt() {
        let mut sender = MultipathSender::new(default_config());
        sender.add_path(make_path(0));
        sender.update_path_rtt(0, Duration::from_millis(50));
        let p = sender.paths().first().expect("should have path");
        assert!(p.health.rtt > Duration::ZERO);
    }

    // 19. Usable path count
    #[test]
    fn test_sender_usable_path_count() {
        let mut sender = MultipathSender::new(default_config());
        let mut p0 = make_path(0);
        p0.health.status = PathStatus::Down;
        let p1 = make_path(1);
        sender.add_path(p0);
        sender.add_path(p1);
        assert_eq!(sender.usable_path_count(), 1);
    }

    // 20. Total dispatched counter
    #[test]
    fn test_sender_total_dispatched() {
        let cfg = MultipathConfig::new().with_strategy(SchedulingStrategy::RoundRobin);
        let mut sender = MultipathSender::new(cfg);
        sender.add_path(make_path(0));
        sender.schedule(100);
        sender.schedule(100);
        assert_eq!(sender.total_dispatched(), 2);
    }

    // ── MultipathStreamSender / high-level API tests ──────────────────────────

    fn make_stream_config(scheduler: MultipathScheduler) -> MultipathStreamConfig {
        MultipathStreamConfig::new(
            vec![
                MultipathEndpoint::new("192.168.1.1:5004", 0.5),
                MultipathEndpoint::new("192.168.2.1:5004", 0.5),
            ],
            scheduler,
            RedundancyMode::None,
        )
    }

    // 21. MultipathStreamSender starts with all paths active
    #[test]
    fn test_stream_sender_starts_active() {
        let s = MultipathStreamSender::new(make_stream_config(MultipathScheduler::RoundRobin));
        assert_eq!(s.active_path_count(), 2);
    }

    // 22. RoundRobin alternates between paths
    #[test]
    fn test_stream_sender_round_robin_alternates() {
        let cfg = make_stream_config(MultipathScheduler::RoundRobin);
        let mut s = MultipathStreamSender::new(cfg);
        let p0 = s.select_path();
        let p1 = s.select_path();
        assert_ne!(p0, p1);
        let p2 = s.select_path();
        assert_eq!(p0, p2); // cycles back
    }

    // 23. MinLatency picks path with lower RTT
    #[test]
    fn test_stream_sender_min_latency() {
        let cfg = make_stream_config(MultipathScheduler::MinLatency);
        let mut s = MultipathStreamSender::new(cfg);
        // Give path 0 high RTT, path 1 low RTT
        s.update_path_rtt(0, 150.0);
        s.update_path_rtt(1, 20.0);
        let chosen = s.select_path();
        assert_eq!(chosen, 1, "should prefer path with lower RTT");
    }

    // 24. WeightedRoundRobin selects in weight order
    #[test]
    fn test_stream_sender_weighted_round_robin() {
        let cfg = MultipathStreamConfig::new(
            vec![
                MultipathEndpoint::new("10.0.0.1:5004", 0.8),
                MultipathEndpoint::new("10.0.0.2:5004", 0.2),
            ],
            MultipathScheduler::WeightedRoundRobin,
            RedundancyMode::None,
        );
        let mut s = MultipathStreamSender::new(cfg);
        // Path 0 has higher weight so should be selected first
        let chosen = s.select_path();
        assert_eq!(chosen, 0);
    }

    // 25. Redundant scheduler returns first active path from select_path
    #[test]
    fn test_stream_sender_redundant_returns_index() {
        let cfg = make_stream_config(MultipathScheduler::Redundant);
        let mut s = MultipathStreamSender::new(cfg);
        let idx = s.select_path();
        assert!(idx < 2);
    }

    // 26. active_path_indices returns all active paths
    #[test]
    fn test_stream_sender_active_path_indices() {
        let cfg = make_stream_config(MultipathScheduler::Redundant);
        let s = MultipathStreamSender::new(cfg);
        let indices = s.active_path_indices();
        assert_eq!(indices, vec![0, 1]);
    }

    // 27. set_path_active deactivates a path
    #[test]
    fn test_stream_sender_deactivate_path() {
        let cfg = make_stream_config(MultipathScheduler::RoundRobin);
        let mut s = MultipathStreamSender::new(cfg);
        s.set_path_active(0, false);
        assert_eq!(s.active_path_count(), 1);
        let idx = s.select_path();
        assert_eq!(idx, 1); // only path 1 is active
    }

    // 28. update_path_rtt stores value via EWMA
    #[test]
    fn test_stream_sender_update_rtt() {
        let cfg = make_stream_config(MultipathScheduler::RoundRobin);
        let mut s = MultipathStreamSender::new(cfg);
        s.update_path_rtt(0, 80.0);
        assert!(s.stats().path_stats[0].rtt_ms > 0.0);
    }

    // 29. update_path_loss marks path inactive at high loss
    #[test]
    fn test_stream_sender_high_loss_deactivates_path() {
        let cfg = make_stream_config(MultipathScheduler::RoundRobin);
        let mut s = MultipathStreamSender::new(cfg);
        // Apply 100 % loss many times to converge the EWMA above 90 %
        for _ in 0..40 {
            s.update_path_loss(0, 1.0);
        }
        assert!(!s.stats().path_stats[0].active);
    }

    // 30. record_bytes_sent accumulates in path and aggregate stats
    #[test]
    fn test_stream_sender_record_bytes() {
        let cfg = make_stream_config(MultipathScheduler::RoundRobin);
        let mut s = MultipathStreamSender::new(cfg);
        s.record_bytes_sent(0, 1000);
        s.record_bytes_sent(1, 500);
        assert_eq!(s.stats().path_stats[0].bytes_sent, 1000);
        assert_eq!(s.stats().total_bytes_sent, 1500);
    }

    // 31. PathStats::new_active initialises to active
    #[test]
    fn test_path_stats_new_active() {
        let ps = PathStats::new_active();
        assert!(ps.active);
        assert_eq!(ps.rtt_ms, 0.0);
        assert_eq!(ps.packet_loss_rate, 0.0);
    }

    // 32. MultipathEndpoint::with_interface attaches NIC name
    #[test]
    fn test_endpoint_with_interface() {
        let ep = MultipathEndpoint::new("10.0.0.1:5004", 1.0).with_interface("eth0");
        assert_eq!(ep.interface.as_deref(), Some("eth0"));
    }
}
