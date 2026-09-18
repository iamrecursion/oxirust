//! Flow statistics tracking for `VideoIP` streams.
//!
//! Provides per-flow metrics including bitrate, packet loss rate, and jitter.

#![allow(dead_code)]

/// Unit of measurement for a flow metric.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowMetric {
    /// Bits per second.
    Bitrate,
    /// Percentage of packets lost (0–100).
    PacketLoss,
    /// Jitter measured in milliseconds.
    JitterMs,
    /// Round-trip time in milliseconds.
    RttMs,
}

impl FlowMetric {
    /// Returns the unit string for this metric.
    #[must_use]
    pub fn unit(&self) -> &'static str {
        match self {
            FlowMetric::Bitrate => "bps",
            FlowMetric::PacketLoss => "%",
            FlowMetric::JitterMs => "ms",
            FlowMetric::RttMs => "ms",
        }
    }
}

/// Accumulates per-flow statistics for a `VideoIP` stream.
#[derive(Debug, Clone)]
pub struct FlowStats {
    /// Total bytes observed.
    total_bytes: u64,
    /// Elapsed time in seconds over which bytes were accumulated.
    elapsed_secs: f64,
    /// Total packets sent (or expected).
    packets_sent: u64,
    /// Packets that were lost.
    packets_lost: u64,
    /// Running sum of inter-packet arrival variation (ms) samples.
    jitter_sum_ms: f64,
    /// Number of jitter samples.
    jitter_samples: u64,
}

impl FlowStats {
    /// Creates a new empty `FlowStats`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_bytes: 0,
            elapsed_secs: 0.0,
            packets_sent: 0,
            packets_lost: 0,
            jitter_sum_ms: 0.0,
            jitter_samples: 0,
        }
    }

    /// Updates the bitrate accumulator with `bytes` over `secs` seconds.
    pub fn update_bitrate(&mut self, bytes: u64, secs: f64) {
        self.total_bytes += bytes;
        self.elapsed_secs += secs;
    }

    /// Records a packet loss event: `sent` packets expected, `lost` missing.
    pub fn record_packets(&mut self, sent: u64, lost: u64) {
        self.packets_sent += sent;
        self.packets_lost += lost;
    }

    /// Adds a single jitter sample in milliseconds.
    pub fn add_jitter_sample(&mut self, jitter_ms: f64) {
        self.jitter_sum_ms += jitter_ms;
        self.jitter_samples += 1;
    }

    /// Returns the current bitrate estimate in bits per second.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn bitrate_bps(&self) -> f64 {
        if self.elapsed_secs <= 0.0 {
            return 0.0;
        }
        (self.total_bytes as f64 * 8.0) / self.elapsed_secs
    }

    /// Returns the packet loss rate as a value in `[0.0, 1.0]`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn packet_loss_rate(&self) -> f64 {
        if self.packets_sent == 0 {
            return 0.0;
        }
        self.packets_lost as f64 / self.packets_sent as f64
    }

    /// Returns the mean jitter in milliseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn jitter_ms(&self) -> f64 {
        if self.jitter_samples == 0 {
            return 0.0;
        }
        self.jitter_sum_ms / self.jitter_samples as f64
    }

    /// Returns `true` if packet loss exceeds the given threshold (0–1).
    #[must_use]
    pub fn has_excessive_loss(&self, threshold: f64) -> bool {
        self.packet_loss_rate() > threshold
    }
}

impl Default for FlowStats {
    fn default() -> Self {
        Self::new()
    }
}

/// A snapshot report aggregating multiple `FlowStats` entries.
#[derive(Debug, Clone)]
pub struct FlowStatsReport {
    entries: Vec<FlowStats>,
}

impl FlowStatsReport {
    /// Creates an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Appends a `FlowStats` entry to the report.
    pub fn add(&mut self, stats: FlowStats) {
        self.entries.push(stats);
    }

    /// Returns the number of flow entries in the report.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the report contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the `FlowMetric` that is worst across all entries.
    ///
    /// Worst is defined as: highest packet loss first, then highest jitter.
    /// Returns `None` if the report is empty.
    pub fn worst_metric(&self) -> Option<FlowMetric> {
        if self.entries.is_empty() {
            return None;
        }
        let max_loss = self
            .entries
            .iter()
            .map(FlowStats::packet_loss_rate)
            .fold(0.0_f64, f64::max);
        let max_jitter = self
            .entries
            .iter()
            .map(FlowStats::jitter_ms)
            .fold(0.0_f64, f64::max);

        if max_loss >= 0.01 {
            Some(FlowMetric::PacketLoss)
        } else if max_jitter >= 1.0 {
            Some(FlowMetric::JitterMs)
        } else {
            Some(FlowMetric::Bitrate)
        }
    }

    /// Returns the total bitrate across all entries in bps.
    #[must_use]
    pub fn total_bitrate_bps(&self) -> f64 {
        self.entries.iter().map(FlowStats::bitrate_bps).sum()
    }
}

impl Default for FlowStatsReport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_metric_unit_bitrate() {
        assert_eq!(FlowMetric::Bitrate.unit(), "bps");
    }

    #[test]
    fn test_flow_metric_unit_packet_loss() {
        assert_eq!(FlowMetric::PacketLoss.unit(), "%");
    }

    #[test]
    fn test_flow_metric_unit_jitter() {
        assert_eq!(FlowMetric::JitterMs.unit(), "ms");
    }

    #[test]
    fn test_flow_metric_unit_rtt() {
        assert_eq!(FlowMetric::RttMs.unit(), "ms");
    }

    #[test]
    fn test_flow_stats_default_zero() {
        let s = FlowStats::new();
        assert_eq!(s.bitrate_bps(), 0.0);
        assert_eq!(s.packet_loss_rate(), 0.0);
        assert_eq!(s.jitter_ms(), 0.0);
    }

    #[test]
    fn test_update_bitrate() {
        let mut s = FlowStats::new();
        s.update_bitrate(1_000_000, 1.0); // 1 MB in 1 s → 8 Mbps
        assert!((s.bitrate_bps() - 8_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_packet_loss_rate_zero_when_no_loss() {
        let mut s = FlowStats::new();
        s.record_packets(100, 0);
        assert_eq!(s.packet_loss_rate(), 0.0);
    }

    #[test]
    fn test_packet_loss_rate_calculation() {
        let mut s = FlowStats::new();
        s.record_packets(100, 5);
        assert!((s.packet_loss_rate() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_jitter_ms_mean() {
        let mut s = FlowStats::new();
        s.add_jitter_sample(2.0);
        s.add_jitter_sample(4.0);
        assert!((s.jitter_ms() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_has_excessive_loss_true() {
        let mut s = FlowStats::new();
        s.record_packets(100, 10);
        assert!(s.has_excessive_loss(0.05));
    }

    #[test]
    fn test_has_excessive_loss_false() {
        let mut s = FlowStats::new();
        s.record_packets(100, 1);
        assert!(!s.has_excessive_loss(0.05));
    }

    #[test]
    fn test_report_empty() {
        let r = FlowStatsReport::new();
        assert!(r.is_empty());
        assert_eq!(r.worst_metric(), None);
    }

    #[test]
    fn test_report_worst_metric_packet_loss() {
        let mut r = FlowStatsReport::new();
        let mut s = FlowStats::new();
        s.record_packets(100, 5); // 5% loss → worst
        r.add(s);
        assert_eq!(r.worst_metric(), Some(FlowMetric::PacketLoss));
    }

    #[test]
    fn test_report_worst_metric_jitter() {
        let mut r = FlowStatsReport::new();
        let mut s = FlowStats::new();
        s.add_jitter_sample(10.0);
        r.add(s);
        assert_eq!(r.worst_metric(), Some(FlowMetric::JitterMs));
    }

    #[test]
    fn test_report_total_bitrate() {
        let mut r = FlowStatsReport::new();
        let mut s1 = FlowStats::new();
        s1.update_bitrate(125_000, 1.0); // 1 Mbps
        let mut s2 = FlowStats::new();
        s2.update_bitrate(125_000, 1.0); // 1 Mbps
        r.add(s1);
        r.add(s2);
        assert!((r.total_bitrate_bps() - 2_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_flow_stats_default_trait() {
        let s: FlowStats = Default::default();
        assert_eq!(s.bitrate_bps(), 0.0);
    }
}
