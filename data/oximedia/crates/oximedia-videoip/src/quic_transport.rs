//! QUIC-based media transport layer.
//!
//! Provides abstractions for transporting media over QUIC connections,
//! including per-stream prioritisation and connection-level throughput
//! estimation.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Priority of a QUIC stream carrying media data.
///
/// Maps to QUIC urgency levels (lower number = higher urgency).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuicStreamPriority {
    /// Highest priority – signalling and control frames.
    Critical,
    /// High priority – real-time audio/video I-frames.
    High,
    /// Normal priority – P/B frames and secondary streams.
    Normal,
    /// Background – thumbnails, metadata, non-time-critical data.
    Background,
}

impl QuicStreamPriority {
    /// Returns the QUIC urgency value (0 = most urgent, 7 = least).
    #[must_use]
    pub fn urgency(self) -> u8 {
        match self {
            Self::Critical => 0,
            Self::High => 2,
            Self::Normal => 4,
            Self::Background => 6,
        }
    }

    /// Returns `true` if this priority requires real-time delivery.
    #[must_use]
    pub fn is_realtime(self) -> bool {
        matches!(self, Self::Critical | Self::High)
    }

    /// Returns a human-readable label for the priority.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Critical => "critical",
            Self::High => "high",
            Self::Normal => "normal",
            Self::Background => "background",
        }
    }
}

/// A single QUIC stream carrying media data.
#[derive(Debug, Clone)]
pub struct QuicMediaStream {
    /// QUIC stream identifier (client-initiated bidirectional: 0, 4, 8, …).
    pub stream_id: u64,
    /// Stream delivery priority.
    pub priority: QuicStreamPriority,
    /// MIME-like media type string, e.g. `"video/av1"` or `"audio/opus"`.
    pub media_type: String,
    /// Total bytes written to the stream.
    pub bytes_sent: u64,
    /// Bytes confirmed by the remote peer (cumulative ACK offset).
    pub bytes_acked: u64,
}

impl QuicMediaStream {
    /// Creates a new `QuicMediaStream`.
    #[must_use]
    pub fn new(
        stream_id: u64,
        priority: QuicStreamPriority,
        media_type: impl Into<String>,
    ) -> Self {
        Self {
            stream_id,
            priority,
            media_type: media_type.into(),
            bytes_sent: 0,
            bytes_acked: 0,
        }
    }

    /// Bytes sent but not yet acknowledged by the peer.
    #[must_use]
    pub fn unacknowledged(&self) -> u64 {
        self.bytes_sent.saturating_sub(self.bytes_acked)
    }

    /// Fraction of sent data acknowledged by the peer, in percent.
    ///
    /// Returns `100.0` when nothing has been sent yet.
    #[must_use]
    pub fn completion_pct(&self) -> f64 {
        if self.bytes_sent == 0 {
            return 100.0;
        }
        self.bytes_acked as f64 / self.bytes_sent as f64 * 100.0
    }

    /// Returns `true` if all sent data has been acknowledged.
    #[must_use]
    pub fn is_fully_acked(&self) -> bool {
        self.bytes_acked >= self.bytes_sent
    }
}

/// A QUIC connection to a single remote peer.
#[derive(Debug, Clone)]
pub struct QuicConnection {
    /// Remote peer address (e.g. `"192.168.1.10:4433"`).
    pub peer_addr: String,
    /// Smoothed round-trip time in milliseconds.
    pub rtt_ms: f32,
    /// Current congestion window in bytes.
    pub cwnd_bytes: u64,
    /// Active media streams on this connection.
    pub streams: Vec<QuicMediaStream>,
}

impl QuicConnection {
    /// Creates a new `QuicConnection`.
    #[must_use]
    pub fn new(peer_addr: impl Into<String>, rtt_ms: f32, cwnd_bytes: u64) -> Self {
        Self {
            peer_addr: peer_addr.into(),
            rtt_ms,
            cwnd_bytes,
            streams: Vec::new(),
        }
    }

    /// Adds a stream to this connection.
    pub fn add_stream(&mut self, stream: QuicMediaStream) {
        self.streams.push(stream);
    }

    /// Returns references to all active streams.
    #[must_use]
    pub fn active_streams(&self) -> Vec<&QuicMediaStream> {
        self.streams.iter().collect()
    }

    /// Returns references to streams at or above the given priority.
    #[must_use]
    pub fn streams_with_priority(&self, min: QuicStreamPriority) -> Vec<&QuicMediaStream> {
        self.streams.iter().filter(|s| s.priority <= min).collect()
    }

    /// Estimates the achievable throughput in bits per second.
    ///
    /// Uses the simplified formula: `cwnd / rtt`.
    /// Returns `0.0` if RTT is zero.
    #[must_use]
    pub fn throughput_estimate_bps(&self) -> f64 {
        if self.rtt_ms <= 0.0 {
            return 0.0;
        }
        let rtt_s = f64::from(self.rtt_ms) / 1_000.0;
        self.cwnd_bytes as f64 * 8.0 / rtt_s
    }

    /// Total unacknowledged bytes across all streams.
    #[must_use]
    pub fn total_unacked(&self) -> u64 {
        self.streams
            .iter()
            .map(QuicMediaStream::unacknowledged)
            .sum()
    }
}

/// Configuration for a QUIC transport endpoint.
#[derive(Debug, Clone)]
pub struct QuicTransportConfig {
    /// Maximum number of concurrently open bidirectional streams.
    pub initial_max_streams: u64,
    /// Maximum total data (bytes) the remote may send before flow-control blocks.
    pub initial_max_data: u64,
    /// Idle connection timeout in milliseconds.
    pub idle_timeout_ms: u64,
    /// Keep-alive ping interval in milliseconds (0 = disabled).
    pub keep_alive_ms: u64,
}

impl QuicTransportConfig {
    /// Creates a new `QuicTransportConfig`.
    #[must_use]
    pub fn new(
        initial_max_streams: u64,
        initial_max_data: u64,
        idle_timeout_ms: u64,
        keep_alive_ms: u64,
    ) -> Self {
        Self {
            initial_max_streams,
            initial_max_data,
            idle_timeout_ms,
            keep_alive_ms,
        }
    }

    /// Validates that the configuration is internally consistent.
    ///
    /// Returns `false` if any value is obviously wrong (e.g. zero streams,
    /// keep-alive >= idle timeout).
    #[must_use]
    pub fn validate(&self) -> bool {
        if self.initial_max_streams == 0 {
            return false;
        }
        if self.initial_max_data == 0 {
            return false;
        }
        if self.idle_timeout_ms == 0 {
            return false;
        }
        if self.keep_alive_ms > 0 && self.keep_alive_ms >= self.idle_timeout_ms {
            return false;
        }
        true
    }

    /// Returns a sensible default for low-latency media streaming.
    #[must_use]
    pub fn default_media() -> Self {
        Self::new(
            100,              // max streams
            16 * 1024 * 1024, // 16 MiB initial max data
            30_000,           // 30 s idle timeout
            5_000,            // 5 s keep-alive
        )
    }
}

impl Default for QuicTransportConfig {
    fn default() -> Self {
        Self::default_media()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. QuicStreamPriority::urgency ordering
    #[test]
    fn test_urgency_ordering() {
        assert!(QuicStreamPriority::Critical.urgency() < QuicStreamPriority::High.urgency());
        assert!(QuicStreamPriority::High.urgency() < QuicStreamPriority::Normal.urgency());
        assert!(QuicStreamPriority::Normal.urgency() < QuicStreamPriority::Background.urgency());
    }

    // 2. QuicStreamPriority::is_realtime
    #[test]
    fn test_is_realtime() {
        assert!(QuicStreamPriority::Critical.is_realtime());
        assert!(QuicStreamPriority::High.is_realtime());
        assert!(!QuicStreamPriority::Normal.is_realtime());
        assert!(!QuicStreamPriority::Background.is_realtime());
    }

    // 3. QuicStreamPriority::label
    #[test]
    fn test_priority_labels() {
        assert_eq!(QuicStreamPriority::Critical.label(), "critical");
        assert_eq!(QuicStreamPriority::Background.label(), "background");
    }

    // 4. QuicMediaStream::unacknowledged
    #[test]
    fn test_unacknowledged_bytes() {
        let mut s = QuicMediaStream::new(0, QuicStreamPriority::High, "video/av1");
        s.bytes_sent = 1000;
        s.bytes_acked = 700;
        assert_eq!(s.unacknowledged(), 300);
    }

    // 5. QuicMediaStream::completion_pct – nothing sent
    #[test]
    fn test_completion_pct_nothing_sent() {
        let s = QuicMediaStream::new(0, QuicStreamPriority::Normal, "audio/opus");
        assert!((s.completion_pct() - 100.0).abs() < 1e-6);
    }

    // 6. QuicMediaStream::completion_pct – partial
    #[test]
    fn test_completion_pct_partial() {
        let mut s = QuicMediaStream::new(4, QuicStreamPriority::Normal, "video/vp9");
        s.bytes_sent = 200;
        s.bytes_acked = 100;
        assert!((s.completion_pct() - 50.0).abs() < 1e-6);
    }

    // 7. QuicMediaStream::is_fully_acked
    #[test]
    fn test_is_fully_acked_true() {
        let mut s = QuicMediaStream::new(8, QuicStreamPriority::Critical, "meta");
        s.bytes_sent = 512;
        s.bytes_acked = 512;
        assert!(s.is_fully_acked());
    }

    #[test]
    fn test_is_fully_acked_false() {
        let mut s = QuicMediaStream::new(8, QuicStreamPriority::Critical, "meta");
        s.bytes_sent = 512;
        s.bytes_acked = 256;
        assert!(!s.is_fully_acked());
    }

    // 8. QuicConnection::throughput_estimate_bps – zero RTT
    #[test]
    fn test_throughput_zero_rtt() {
        let conn = QuicConnection::new("10.0.0.1:4433", 0.0, 1_000_000);
        assert_eq!(conn.throughput_estimate_bps(), 0.0);
    }

    // 9. QuicConnection::throughput_estimate_bps – normal
    #[test]
    fn test_throughput_estimate() {
        // cwnd = 125_000 bytes, rtt = 10 ms → bps = 125_000 * 8 / 0.01 = 100_000_000
        let conn = QuicConnection::new("10.0.0.1:4433", 10.0, 125_000);
        assert!((conn.throughput_estimate_bps() - 100_000_000.0).abs() < 1.0);
    }

    // 10. QuicConnection::add_stream and active_streams
    #[test]
    fn test_connection_streams() {
        let mut conn = QuicConnection::new("127.0.0.1:4433", 5.0, 65536);
        assert!(conn.active_streams().is_empty());
        conn.add_stream(QuicMediaStream::new(
            0,
            QuicStreamPriority::High,
            "video/av1",
        ));
        conn.add_stream(QuicMediaStream::new(
            4,
            QuicStreamPriority::Normal,
            "audio/opus",
        ));
        assert_eq!(conn.active_streams().len(), 2);
    }

    // 11. QuicConnection::streams_with_priority
    #[test]
    fn test_streams_with_priority_filter() {
        let mut conn = QuicConnection::new("127.0.0.1:4433", 5.0, 65536);
        conn.add_stream(QuicMediaStream::new(
            0,
            QuicStreamPriority::Critical,
            "ctrl",
        ));
        conn.add_stream(QuicMediaStream::new(
            4,
            QuicStreamPriority::High,
            "video/av1",
        ));
        conn.add_stream(QuicMediaStream::new(
            8,
            QuicStreamPriority::Background,
            "thumb",
        ));
        // Streams at or above Normal (Critical, High, Normal) = 2
        let high_plus = conn.streams_with_priority(QuicStreamPriority::High);
        assert_eq!(high_plus.len(), 2);
    }

    // 12. QuicConnection::total_unacked
    #[test]
    fn test_total_unacked() {
        let mut conn = QuicConnection::new("127.0.0.1:4433", 5.0, 65536);
        let mut s1 = QuicMediaStream::new(0, QuicStreamPriority::High, "video/av1");
        s1.bytes_sent = 1000;
        s1.bytes_acked = 800;
        let mut s2 = QuicMediaStream::new(4, QuicStreamPriority::Normal, "audio/opus");
        s2.bytes_sent = 500;
        s2.bytes_acked = 200;
        conn.add_stream(s1);
        conn.add_stream(s2);
        assert_eq!(conn.total_unacked(), 500);
    }

    // 13. QuicTransportConfig::validate – valid config
    #[test]
    fn test_config_valid() {
        let cfg = QuicTransportConfig::default_media();
        assert!(cfg.validate());
    }

    // 14. QuicTransportConfig::validate – zero streams
    #[test]
    fn test_config_invalid_zero_streams() {
        let cfg = QuicTransportConfig::new(0, 1024, 30_000, 5_000);
        assert!(!cfg.validate());
    }

    // 15. QuicTransportConfig::validate – keep-alive >= idle timeout
    #[test]
    fn test_config_invalid_keepalive() {
        let cfg = QuicTransportConfig::new(10, 1024, 5_000, 5_000);
        assert!(!cfg.validate());
    }
}
