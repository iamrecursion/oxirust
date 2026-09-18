//! SRT stream statistics and quality monitoring.

#![allow(dead_code)]

use std::time::Instant;

/// Per-direction statistics.
#[derive(Debug, Clone, Default)]
pub struct DirectionStats {
    /// Total packets sent or expected.
    pub packets_sent: u64,
    /// Total packets received.
    pub packets_received: u64,
    /// Packets lost (not received).
    pub packets_lost: u64,
    /// Packets retransmitted.
    pub packets_retransmitted: u64,
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Packet loss rate in [0.0, 1.0].
    pub packet_loss_rate: f64,
    /// Retransmit rate in [0.0, 1.0].
    pub retransmit_rate: f64,
    /// Estimated bandwidth in bits per second.
    pub bandwidth_bps: f64,
}

impl DirectionStats {
    /// Recalculate `packet_loss_rate` from packet counts.
    pub fn update_loss_rate(&mut self) {
        let total = self.packets_sent;
        if total == 0 {
            self.packet_loss_rate = 0.0;
        } else {
            self.packet_loss_rate = self.packets_lost as f64 / total as f64;
        }
    }

    /// Recalculate `retransmit_rate` from packet counts.
    pub fn update_retransmit_rate(&mut self) {
        let total = self.packets_sent;
        if total == 0 {
            self.retransmit_rate = 0.0;
        } else {
            self.retransmit_rate = self.packets_retransmitted as f64 / total as f64;
        }
    }
}

/// RTT (Round-Trip Time) statistics using Welford's online algorithm.
#[derive(Debug, Clone)]
pub struct RttStats {
    /// Most recent RTT sample in milliseconds.
    pub current_ms: f64,
    /// Minimum RTT observed.
    pub min_ms: f64,
    /// Maximum RTT observed.
    pub max_ms: f64,
    /// Running mean RTT.
    pub mean_ms: f64,
    /// Running variance of RTT.
    pub variance_ms: f64,
    /// Number of samples recorded.
    pub sample_count: u64,
}

impl Default for RttStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RttStats {
    /// Creates a new, empty `RttStats`.
    pub fn new() -> Self {
        Self {
            current_ms: 0.0,
            min_ms: f64::MAX,
            max_ms: 0.0,
            mean_ms: 0.0,
            variance_ms: 0.0,
            sample_count: 0,
        }
    }

    /// Update statistics with a new RTT sample using Welford's online algorithm.
    pub fn update(&mut self, sample_ms: f64) {
        self.current_ms = sample_ms;
        self.sample_count += 1;

        // Update min/max
        if sample_ms < self.min_ms {
            self.min_ms = sample_ms;
        }
        if sample_ms > self.max_ms {
            self.max_ms = sample_ms;
        }

        // Welford's online algorithm for mean and variance
        let n = self.sample_count as f64;
        let delta = sample_ms - self.mean_ms;
        self.mean_ms += delta / n;
        let delta2 = sample_ms - self.mean_ms;
        // M2 accumulator (variance * (n-1))
        self.variance_ms += delta * delta2;
    }

    /// Returns the population variance (variance_ms / sample_count).
    ///
    /// Returns 0.0 if fewer than 2 samples have been recorded.
    pub fn population_variance(&self) -> f64 {
        if self.sample_count < 2 {
            return 0.0;
        }
        self.variance_ms / self.sample_count as f64
    }
}

/// Buffer fill statistics for send and receive buffers.
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    /// Bytes currently in the send buffer.
    pub send_buffer_level: usize,
    /// Bytes currently in the receive buffer.
    pub recv_buffer_level: usize,
    /// Capacity of the send buffer in bytes.
    pub send_buffer_capacity: usize,
    /// Capacity of the receive buffer in bytes.
    pub recv_buffer_capacity: usize,
    /// Send buffer utilization in [0.0, 1.0].
    pub send_buffer_utilization: f64,
    /// Receive buffer utilization in [0.0, 1.0].
    pub recv_buffer_utilization: f64,
}

impl BufferStats {
    /// Recalculate both utilization values from current levels and capacities.
    pub fn update_utilization(&mut self) {
        self.send_buffer_utilization = if self.send_buffer_capacity == 0 {
            0.0
        } else {
            self.send_buffer_level as f64 / self.send_buffer_capacity as f64
        };

        self.recv_buffer_utilization = if self.recv_buffer_capacity == 0 {
            0.0
        } else {
            self.recv_buffer_level as f64 / self.recv_buffer_capacity as f64
        };
    }
}

/// Complete SRT stream statistics snapshot.
#[derive(Debug, Clone)]
pub struct SrtStreamStats {
    /// Send-direction statistics.
    pub send: DirectionStats,
    /// Receive-direction statistics.
    pub recv: DirectionStats,
    /// Round-trip time statistics.
    pub rtt: RttStats,
    /// Buffer utilization statistics.
    pub buffer: BufferStats,
    /// Connection uptime in milliseconds (snapshot at creation, use `uptime()` for live).
    pub uptime_ms: u64,
    /// Latency negotiated during the SRT handshake.
    pub negotiated_latency_ms: u32,
    /// Whether AES encryption is active on this stream.
    pub encryption_enabled: bool,
    /// Whether forward-error correction is active on this stream.
    pub fec_enabled: bool,
    /// Timestamp when the connection was established.
    pub connected_at: Instant,
}

impl SrtStreamStats {
    /// Create a new statistics object for a connection with the given negotiated latency.
    pub fn new(latency_ms: u32) -> Self {
        Self {
            send: DirectionStats::default(),
            recv: DirectionStats::default(),
            rtt: RttStats::new(),
            buffer: BufferStats::default(),
            uptime_ms: 0,
            negotiated_latency_ms: latency_ms,
            encryption_enabled: false,
            fec_enabled: false,
            connected_at: Instant::now(),
        }
    }

    /// Returns the elapsed time since `connected_at`.
    pub fn uptime(&self) -> std::time::Duration {
        self.connected_at.elapsed()
    }

    /// Returns `true` if the connection is considered healthy:
    /// packet loss < 1 % and RTT < 200 ms.
    pub fn is_healthy(&self) -> bool {
        self.send.packet_loss_rate < 0.01
            && self.recv.packet_loss_rate < 0.01
            && self.rtt.current_ms < 200.0
    }

    /// Returns a quality score in [0.0, 1.0].
    ///
    /// Composite of loss rate, RTT, and buffer utilization.
    /// 1.0 means perfect, 0.0 means worst.
    pub fn quality_score(&self) -> f32 {
        // Loss component: perfect at 0, zero at 10%+ loss
        let loss = self
            .send
            .packet_loss_rate
            .max(self.recv.packet_loss_rate)
            .min(0.1);
        let loss_score = 1.0 - (loss / 0.1);

        // RTT component: perfect at 0 ms, zero at 500 ms+
        let rtt_score = (1.0 - (self.rtt.current_ms / 500.0).min(1.0)).max(0.0);

        // Buffer component: perfect at 0 utilization, zero at 100%
        let buf_util = self
            .buffer
            .send_buffer_utilization
            .max(self.buffer.recv_buffer_utilization)
            .min(1.0);
        let buf_score = 1.0 - buf_util;

        // Weighted average: loss 50%, RTT 35%, buffer 15%
        let score = 0.5 * loss_score + 0.35 * rtt_score + 0.15 * buf_score;
        score.clamp(0.0, 1.0) as f32
    }

    /// Generate a human-readable status report.
    pub fn report(&self) -> String {
        let quality = StreamQuality::from_stats(self);
        format!(
            "SRT Stream Report\n\
             Quality:    {}\n\
             RTT:        {:.1} ms (min={:.1}, max={:.1})\n\
             Loss (tx):  {:.2}%\n\
             Loss (rx):  {:.2}%\n\
             Latency:    {} ms (negotiated)\n\
             Encrypted:  {}\n\
             FEC:        {}\n\
             Uptime:     {:.1}s",
            quality.name(),
            self.rtt.current_ms,
            self.rtt.min_ms.min(self.rtt.max_ms), // guard against MAX sentinel
            self.rtt.max_ms,
            self.send.packet_loss_rate * 100.0,
            self.recv.packet_loss_rate * 100.0,
            self.negotiated_latency_ms,
            self.encryption_enabled,
            self.fec_enabled,
            self.uptime().as_secs_f64(),
        )
    }
}

/// Qualitative assessment of an SRT connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamQuality {
    /// Loss < 0.1 %, RTT < 20 ms.
    Excellent,
    /// Loss < 1 %, RTT < 50 ms.
    Good,
    /// Loss < 5 %, RTT < 100 ms.
    Fair,
    /// Loss < 10 %, RTT < 200 ms.
    Poor,
    /// Worse than Poor.
    Critical,
}

impl StreamQuality {
    /// Derive quality level from a statistics snapshot.
    pub fn from_stats(stats: &SrtStreamStats) -> Self {
        let loss = stats.send.packet_loss_rate.max(stats.recv.packet_loss_rate);
        let rtt = stats.rtt.current_ms;

        if loss < 0.001 && rtt < 20.0 {
            Self::Excellent
        } else if loss < 0.01 && rtt < 50.0 {
            Self::Good
        } else if loss < 0.05 && rtt < 100.0 {
            Self::Fair
        } else if loss < 0.1 && rtt < 200.0 {
            Self::Poor
        } else {
            Self::Critical
        }
    }

    /// Short human-readable name for the quality level.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::Poor => "Poor",
            Self::Critical => "Critical",
        }
    }

    /// Returns `true` for any quality level that can sustain a usable stream
    /// (Excellent through Poor).
    pub fn is_usable(&self) -> bool {
        !matches!(self, Self::Critical)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtt_stats_update() {
        let mut rtt = RttStats::new();
        rtt.update(10.0);
        rtt.update(20.0);
        rtt.update(30.0);
        // Mean should be approximately 20.0
        assert!((rtt.mean_ms - 20.0).abs() < 1e-9, "mean={}", rtt.mean_ms);
        assert_eq!(rtt.sample_count, 3);
        assert_eq!(rtt.current_ms, 30.0);
    }

    #[test]
    fn test_rtt_stats_min_max() {
        let mut rtt = RttStats::new();
        rtt.update(50.0);
        rtt.update(10.0);
        rtt.update(100.0);
        assert_eq!(rtt.min_ms, 10.0);
        assert_eq!(rtt.max_ms, 100.0);
    }

    #[test]
    fn test_direction_stats_loss_rate() {
        let mut d = DirectionStats::default();
        d.packets_sent = 100;
        d.packets_lost = 10;
        d.update_loss_rate();
        assert!((d.packet_loss_rate - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_buffer_stats_utilization() {
        let mut b = BufferStats::default();
        b.send_buffer_level = 512;
        b.send_buffer_capacity = 1024;
        b.recv_buffer_level = 0;
        b.recv_buffer_capacity = 1024;
        b.update_utilization();
        assert!((b.send_buffer_utilization - 0.5).abs() < 1e-9);
        assert!((b.recv_buffer_utilization - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_stream_stats_healthy() {
        let stats = SrtStreamStats::new(120);
        // Default: all zeros → healthy
        assert!(stats.is_healthy());
    }

    #[test]
    fn test_stream_quality_from_stats() {
        let mut stats = SrtStreamStats::new(120);
        stats.send.packet_loss_rate = 0.005; // 0.5%
        stats.rtt.current_ms = 30.0;
        let quality = StreamQuality::from_stats(&stats);
        assert_eq!(quality, StreamQuality::Good);
    }

    #[test]
    fn test_stream_quality_is_usable() {
        assert!(StreamQuality::Excellent.is_usable());
        assert!(StreamQuality::Good.is_usable());
        assert!(StreamQuality::Fair.is_usable());
        assert!(StreamQuality::Poor.is_usable());
        assert!(!StreamQuality::Critical.is_usable());
    }

    #[test]
    fn test_quality_score_range() {
        // All-zero stats → near-perfect score
        let mut stats = SrtStreamStats::new(120);
        let score = stats.quality_score();
        assert!((0.0..=1.0).contains(&score));

        // Worst-case stats
        stats.send.packet_loss_rate = 1.0;
        stats.recv.packet_loss_rate = 1.0;
        stats.rtt.current_ms = 1000.0;
        stats.buffer.send_buffer_utilization = 1.0;
        stats.buffer.recv_buffer_utilization = 1.0;
        let score_bad = stats.quality_score();
        assert!((0.0..=1.0).contains(&score_bad));
        assert!(score > score_bad);
    }
}
