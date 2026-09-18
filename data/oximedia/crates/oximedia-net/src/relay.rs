//! Media relay and restreaming server.
//!
//! A media relay receives a stream on one protocol and retransmits it on
//! another protocol (or the same protocol to multiple destinations).
//!
//! Supported relay scenarios:
//! - **RTMP → RTMP**: receive an RTMP publish, re-publish to multiple RTMP endpoints.
//! - **RTMP → HLS**: receive RTMP, produce an HLS playlist.
//! - **SRT → RTMP**: bridge SRT to RTMP.
//! - **SRT → SRT**: fan-out a single SRT input to multiple SRT receivers.
//! - **HLS → HLS**: pull-and-relay HLS playlists.
//!
//! Design:
//! - [`RelayProtocol`] — identifies an input/output protocol.
//! - [`RelayEndpoint`] — an address + protocol combination (source or destination).
//! - [`RelayRoute`] — maps one input endpoint to one or more output endpoints.
//! - [`RelayServer`] — manages a set of routes and tracks metrics.
//! - [`StreamRelay`] — in-process relay that copies data frames between channels.

#![allow(dead_code)]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

// ─── Protocol ─────────────────────────────────────────────────────────────────

/// Streaming protocol identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelayProtocol {
    /// Real-Time Messaging Protocol.
    Rtmp,
    /// Secure Reliable Transport.
    Srt,
    /// HTTP Live Streaming (pull-based).
    Hls,
    /// MPEG-DASH (pull-based).
    Dash,
    /// RIST (Reliable Internet Stream Transport).
    Rist,
    /// WebRTC (WHIP/WHEP).
    WebRtc,
    /// Raw UDP/RTP stream.
    Rtp,
}

impl RelayProtocol {
    /// Returns the protocol name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Rtmp => "RTMP",
            Self::Srt => "SRT",
            Self::Hls => "HLS",
            Self::Dash => "DASH",
            Self::Rist => "RIST",
            Self::WebRtc => "WebRTC",
            Self::Rtp => "RTP",
        }
    }

    /// Returns the default port for this protocol.
    #[must_use]
    pub const fn default_port(&self) -> u16 {
        match self {
            Self::Rtmp => 1935,
            Self::Srt => 9000,
            Self::Hls => 80,
            Self::Dash => 80,
            Self::Rist => 5004,
            Self::WebRtc => 443,
            Self::Rtp => 5004,
        }
    }

    /// Returns whether the protocol is push-based (sender initiates).
    #[must_use]
    pub const fn is_push(&self) -> bool {
        matches!(self, Self::Rtmp | Self::Srt | Self::Rist | Self::Rtp)
    }
}

// ─── Endpoint ─────────────────────────────────────────────────────────────────

/// A relay endpoint combining a protocol and transport address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelayEndpoint {
    /// Protocol used on this endpoint.
    pub protocol: RelayProtocol,
    /// Network address.
    pub address: SocketAddr,
    /// Optional stream key or path (e.g. RTMP stream key, HLS playlist path).
    pub stream_key: Option<String>,
    /// Optional passphrase (for SRT encryption).
    pub passphrase: Option<String>,
}

impl RelayEndpoint {
    /// Creates a new relay endpoint.
    #[must_use]
    pub fn new(protocol: RelayProtocol, address: SocketAddr) -> Self {
        Self {
            protocol,
            address,
            stream_key: None,
            passphrase: None,
        }
    }

    /// Sets the stream key.
    #[must_use]
    pub fn with_stream_key(mut self, key: impl Into<String>) -> Self {
        self.stream_key = Some(key.into());
        self
    }

    /// Sets the passphrase.
    #[must_use]
    pub fn with_passphrase(mut self, pass: impl Into<String>) -> Self {
        self.passphrase = Some(pass.into());
        self
    }

    /// Returns a URL-like representation.
    #[must_use]
    pub fn to_url(&self) -> String {
        let proto = self.protocol.name().to_lowercase();
        let base = format!("{}://{}", proto, self.address);
        match &self.stream_key {
            Some(k) => format!("{}/{}", base, k),
            None => base,
        }
    }
}

// ─── Route Status ─────────────────────────────────────────────────────────────

/// Operational status of a relay route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteStatus {
    /// Route is active and streaming.
    Active,
    /// Route is configured but not yet connected.
    Idle,
    /// Route is reconnecting after a failure.
    Reconnecting,
    /// Route encountered a fatal error.
    Error,
    /// Route has been stopped.
    Stopped,
}

// ─── Route Statistics ─────────────────────────────────────────────────────────

/// Statistics for a relay route.
#[derive(Debug, Clone, Default)]
pub struct RouteStats {
    /// Total bytes relayed.
    pub bytes_relayed: u64,
    /// Total frames relayed.
    pub frames_relayed: u64,
    /// Total frames dropped.
    pub frames_dropped: u64,
    /// Number of reconnections.
    pub reconnection_count: u32,
    /// Current input bitrate in bits per second.
    pub input_bitrate_bps: f64,
    /// Current output bitrate in bits per second.
    pub output_bitrate_bps: f64,
    /// Average relay latency.
    pub avg_relay_latency: Duration,
}

// ─── Relay Route ──────────────────────────────────────────────────────────────

/// A relay route: one input endpoint → one or more output endpoints.
#[derive(Debug)]
pub struct RelayRoute {
    /// Unique route identifier.
    pub id: u32,
    /// Input (source) endpoint.
    pub input: RelayEndpoint,
    /// Output (destination) endpoints.
    pub outputs: Vec<RelayEndpoint>,
    /// Current status.
    pub status: RouteStatus,
    /// Route statistics.
    pub stats: RouteStats,
    /// When the route was created.
    pub created_at: Instant,
    /// When streaming started.
    pub stream_start: Option<Instant>,
    /// Maximum output queue depth before frames are dropped.
    pub max_queue_depth: usize,
    /// Whether to buffer frames on output disconnects (or drop).
    pub buffer_on_disconnect: bool,
}

impl RelayRoute {
    /// Creates a new relay route.
    #[must_use]
    pub fn new(id: u32, input: RelayEndpoint) -> Self {
        Self {
            id,
            input,
            outputs: Vec::new(),
            status: RouteStatus::Idle,
            stats: RouteStats::default(),
            created_at: Instant::now(),
            stream_start: None,
            max_queue_depth: 120, // 2 seconds at 60fps
            buffer_on_disconnect: true,
        }
    }

    /// Adds an output endpoint.
    pub fn add_output(&mut self, output: RelayEndpoint) {
        self.outputs.push(output);
    }

    /// Removes an output endpoint by address.
    pub fn remove_output(&mut self, addr: SocketAddr) {
        self.outputs.retain(|o| o.address != addr);
    }

    /// Returns the number of output endpoints.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Marks the route as active and records the stream start time.
    pub fn mark_active(&mut self) {
        self.status = RouteStatus::Active;
        self.stream_start = Some(Instant::now());
    }

    /// Returns the elapsed stream time.
    #[must_use]
    pub fn stream_duration(&self) -> Option<Duration> {
        self.stream_start.map(|t| t.elapsed())
    }

    /// Simulates relaying a frame of `frame_bytes` bytes.
    ///
    /// Updates statistics; returns the number of outputs the frame was sent to.
    pub fn relay_frame(&mut self, frame_bytes: usize) -> usize {
        if self.status != RouteStatus::Active {
            self.stats.frames_dropped += 1;
            return 0;
        }
        let count = self.outputs.len();
        self.stats.bytes_relayed += frame_bytes as u64 * count as u64;
        self.stats.frames_relayed += 1;
        count
    }

    /// Returns the ratio of dropped frames.
    #[must_use]
    pub fn drop_ratio(&self) -> f64 {
        let total = self.stats.frames_relayed + self.stats.frames_dropped;
        if total == 0 {
            return 0.0;
        }
        self.stats.frames_dropped as f64 / total as f64
    }
}

// ─── Protocol Transcoder Hint ─────────────────────────────────────────────────

/// Describes the translation required at a protocol boundary.
#[derive(Debug, Clone)]
pub struct ProtocolBridge {
    /// Source protocol.
    pub from: RelayProtocol,
    /// Destination protocol.
    pub to: RelayProtocol,
    /// Whether transcoding (reencoding) is needed (vs. mux-only passthrough).
    pub requires_transcode: bool,
}

impl ProtocolBridge {
    /// Creates a new protocol bridge.
    #[must_use]
    pub fn new(from: RelayProtocol, to: RelayProtocol) -> Self {
        // Transcode is needed when the container changes significantly.
        let requires_transcode = matches!(
            (from, to),
            (RelayProtocol::Rtmp, RelayProtocol::Hls)
                | (RelayProtocol::Rtmp, RelayProtocol::Dash)
                | (RelayProtocol::Srt, RelayProtocol::Hls)
                | (RelayProtocol::Srt, RelayProtocol::Dash)
                | (RelayProtocol::WebRtc, RelayProtocol::Rtmp)
                | (RelayProtocol::WebRtc, RelayProtocol::Hls)
        );
        Self {
            from,
            to,
            requires_transcode,
        }
    }

    /// Returns whether this bridge is a passthrough (same protocol).
    #[must_use]
    pub fn is_passthrough(&self) -> bool {
        self.from == self.to
    }
}

// ─── Relay Server ─────────────────────────────────────────────────────────────

/// Media relay server managing multiple routes.
pub struct RelayServer {
    /// All configured routes.
    routes: HashMap<u32, RelayRoute>,
    /// Next route ID.
    next_id: u32,
    /// Maximum total routes.
    max_routes: usize,
    /// Total frames processed.
    total_frames: u64,
}

impl RelayServer {
    /// Creates a new relay server.
    #[must_use]
    pub fn new(max_routes: usize) -> Self {
        Self {
            routes: HashMap::new(),
            next_id: 1,
            max_routes,
            total_frames: 0,
        }
    }

    /// Adds a relay route.
    ///
    /// Returns the route ID or `None` if the server is full.
    pub fn add_route(&mut self, input: RelayEndpoint) -> Option<u32> {
        if self.routes.len() >= self.max_routes {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.routes.insert(id, RelayRoute::new(id, input));
        Some(id)
    }

    /// Removes a route by ID.
    pub fn remove_route(&mut self, id: u32) -> bool {
        self.routes.remove(&id).is_some()
    }

    /// Returns a reference to a route.
    #[must_use]
    pub fn route(&self, id: u32) -> Option<&RelayRoute> {
        self.routes.get(&id)
    }

    /// Returns a mutable reference to a route.
    pub fn route_mut(&mut self, id: u32) -> Option<&mut RelayRoute> {
        self.routes.get_mut(&id)
    }

    /// Returns the number of active routes.
    #[must_use]
    pub fn active_route_count(&self) -> usize {
        self.routes
            .values()
            .filter(|r| r.status == RouteStatus::Active)
            .count()
    }

    /// Returns the total number of routes.
    #[must_use]
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// Processes a frame through all active routes.
    ///
    /// Returns the total number of output deliveries.
    pub fn process_frame(&mut self, route_id: u32, frame_bytes: usize) -> usize {
        if let Some(route) = self.routes.get_mut(&route_id) {
            self.total_frames += 1;
            route.relay_frame(frame_bytes)
        } else {
            0
        }
    }

    /// Returns aggregate statistics across all routes.
    #[must_use]
    pub fn aggregate_stats(&self) -> RouteStats {
        let mut agg = RouteStats::default();
        for r in self.routes.values() {
            agg.bytes_relayed += r.stats.bytes_relayed;
            agg.frames_relayed += r.stats.frames_relayed;
            agg.frames_dropped += r.stats.frames_dropped;
            agg.reconnection_count += r.stats.reconnection_count;
        }
        agg
    }
}

// ─── Stream Relay ─────────────────────────────────────────────────────────────

/// In-process stream relay: copies data between two abstract channels.
///
/// This is a testing/integration aid.  Real protocol I/O is handled by the
/// protocol-specific modules; this relay just passes [`RelayFrame`]s between
/// them.
#[derive(Debug, Clone)]
pub struct RelayFrame {
    /// Frame sequence number.
    pub seq: u64,
    /// Frame timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Frame data.
    pub data: Vec<u8>,
    /// Whether this is a keyframe.
    pub is_keyframe: bool,
}

/// A simple in-process relay backed by channels (modelled as queues here).
pub struct StreamRelay {
    /// Input queue (from the source).
    input_queue: std::collections::VecDeque<RelayFrame>,
    /// Output queue (to the destination).
    output_queue: std::collections::VecDeque<RelayFrame>,
    /// Maximum queue depth.
    max_depth: usize,
    /// Frames relayed.
    frames_relayed: u64,
    /// Frames dropped.
    frames_dropped: u64,
    /// Source protocol.
    source_protocol: RelayProtocol,
    /// Destination protocol.
    dest_protocol: RelayProtocol,
}

impl StreamRelay {
    /// Creates a new stream relay.
    #[must_use]
    pub fn new(
        source_protocol: RelayProtocol,
        dest_protocol: RelayProtocol,
        max_depth: usize,
    ) -> Self {
        Self {
            input_queue: std::collections::VecDeque::new(),
            output_queue: std::collections::VecDeque::new(),
            max_depth,
            frames_relayed: 0,
            frames_dropped: 0,
            source_protocol,
            dest_protocol,
        }
    }

    /// Pushes a frame from the source into the relay.
    pub fn push_input(&mut self, frame: RelayFrame) {
        if self.input_queue.len() >= self.max_depth {
            self.frames_dropped += 1;
            return;
        }
        self.input_queue.push_back(frame);
    }

    /// Processes all queued input frames and moves them to the output queue.
    pub fn process(&mut self) {
        while let Some(frame) = self.input_queue.pop_front() {
            if self.output_queue.len() < self.max_depth {
                self.output_queue.push_back(frame);
                self.frames_relayed += 1;
            } else {
                self.frames_dropped += 1;
            }
        }
    }

    /// Pops the next output frame.
    pub fn pop_output(&mut self) -> Option<RelayFrame> {
        self.output_queue.pop_front()
    }

    /// Returns the protocol bridge description.
    #[must_use]
    pub fn protocol_bridge(&self) -> ProtocolBridge {
        ProtocolBridge::new(self.source_protocol, self.dest_protocol)
    }

    /// Returns frames relayed.
    #[must_use]
    pub const fn frames_relayed(&self) -> u64 {
        self.frames_relayed
    }

    /// Returns frames dropped.
    #[must_use]
    pub const fn frames_dropped(&self) -> u64 {
        self.frames_dropped
    }

    /// Returns current input queue depth.
    #[must_use]
    pub fn input_depth(&self) -> usize {
        self.input_queue.len()
    }

    /// Returns current output queue depth.
    #[must_use]
    pub fn output_depth(&self) -> usize {
        self.output_queue.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rtmp_endpoint() -> RelayEndpoint {
        RelayEndpoint::new(
            RelayProtocol::Rtmp,
            "127.0.0.1:1935".parse().expect("valid addr"),
        )
        .with_stream_key("live/stream1")
    }

    fn make_srt_endpoint() -> RelayEndpoint {
        RelayEndpoint::new(
            RelayProtocol::Srt,
            "127.0.0.1:9000".parse().expect("valid addr"),
        )
        .with_passphrase("secret")
    }

    // 1. Protocol names and default ports
    #[test]
    fn test_relay_protocol_names() {
        assert_eq!(RelayProtocol::Rtmp.name(), "RTMP");
        assert_eq!(RelayProtocol::Srt.name(), "SRT");
        assert_eq!(RelayProtocol::Rtmp.default_port(), 1935);
        assert_eq!(RelayProtocol::Srt.default_port(), 9000);
    }

    // 2. Push vs pull
    #[test]
    fn test_relay_protocol_push_pull() {
        assert!(RelayProtocol::Rtmp.is_push());
        assert!(!RelayProtocol::Hls.is_push());
    }

    // 3. Endpoint URL construction
    #[test]
    fn test_endpoint_to_url() {
        let ep = make_rtmp_endpoint();
        let url = ep.to_url();
        assert!(url.contains("rtmp"));
        assert!(url.contains("live/stream1"));
    }

    // 4. RelayRoute creation
    #[test]
    fn test_relay_route_new() {
        let route = RelayRoute::new(1, make_rtmp_endpoint());
        assert_eq!(route.id, 1);
        assert_eq!(route.status, RouteStatus::Idle);
        assert_eq!(route.output_count(), 0);
    }

    // 5. RelayRoute add/remove outputs
    #[test]
    fn test_relay_route_outputs() {
        let mut route = RelayRoute::new(1, make_rtmp_endpoint());
        let addr: SocketAddr = "192.168.1.1:1935".parse().expect("valid addr");
        route.add_output(RelayEndpoint::new(RelayProtocol::Rtmp, addr));
        assert_eq!(route.output_count(), 1);
        route.remove_output(addr);
        assert_eq!(route.output_count(), 0);
    }

    // 6. Relay frame increments stats
    #[test]
    fn test_relay_route_relay_frame() {
        let mut route = RelayRoute::new(1, make_rtmp_endpoint());
        let addr: SocketAddr = "192.168.1.1:1935".parse().expect("valid addr");
        route.add_output(RelayEndpoint::new(RelayProtocol::Rtmp, addr));
        route.mark_active();
        let delivered = route.relay_frame(1000);
        assert_eq!(delivered, 1);
        assert_eq!(route.stats.frames_relayed, 1);
        assert_eq!(route.stats.bytes_relayed, 1000);
    }

    // 7. Idle route drops frames
    #[test]
    fn test_relay_route_idle_drops_frames() {
        let mut route = RelayRoute::new(1, make_rtmp_endpoint());
        route.relay_frame(100); // Not active
        assert_eq!(route.stats.frames_dropped, 1);
    }

    // 8. Drop ratio
    #[test]
    fn test_relay_route_drop_ratio() {
        let route = RelayRoute::new(1, make_rtmp_endpoint());
        assert_eq!(route.drop_ratio(), 0.0);
    }

    // 9. Stream duration
    #[test]
    fn test_relay_route_stream_duration() {
        let mut route = RelayRoute::new(1, make_rtmp_endpoint());
        assert!(route.stream_duration().is_none());
        route.mark_active();
        assert!(route.stream_duration().is_some());
    }

    // 10. RelayServer add and remove route
    #[test]
    fn test_relay_server_add_remove() {
        let mut server = RelayServer::new(10);
        let id = server.add_route(make_rtmp_endpoint()).expect("should add");
        assert_eq!(server.route_count(), 1);
        server.remove_route(id);
        assert_eq!(server.route_count(), 0);
    }

    // 11. RelayServer max routes limit
    #[test]
    fn test_relay_server_max_routes() {
        let mut server = RelayServer::new(2);
        server.add_route(make_rtmp_endpoint());
        server.add_route(make_rtmp_endpoint());
        let result = server.add_route(make_rtmp_endpoint());
        assert!(result.is_none());
    }

    // 12. RelayServer active route count
    #[test]
    fn test_relay_server_active_count() {
        let mut server = RelayServer::new(10);
        let id = server.add_route(make_rtmp_endpoint()).expect("should add");
        server.route_mut(id).expect("should exist").mark_active();
        assert_eq!(server.active_route_count(), 1);
    }

    // 13. RelayServer process_frame
    #[test]
    fn test_relay_server_process_frame() {
        let mut server = RelayServer::new(10);
        let id = server.add_route(make_rtmp_endpoint()).expect("should add");
        let addr: SocketAddr = "192.168.1.1:1935".parse().expect("valid addr");
        {
            let route = server.route_mut(id).expect("should exist");
            route.add_output(RelayEndpoint::new(RelayProtocol::Rtmp, addr));
            route.mark_active();
        }
        let deliveries = server.process_frame(id, 500);
        assert_eq!(deliveries, 1);
    }

    // 14. RelayServer aggregate stats
    #[test]
    fn test_relay_server_aggregate_stats() {
        let mut server = RelayServer::new(10);
        let id = server.add_route(make_rtmp_endpoint()).expect("should add");
        let addr: SocketAddr = "192.168.1.1:1935".parse().expect("valid addr");
        {
            let route = server.route_mut(id).expect("should exist");
            route.add_output(RelayEndpoint::new(RelayProtocol::Rtmp, addr));
            route.mark_active();
        }
        server.process_frame(id, 100);
        let stats = server.aggregate_stats();
        assert_eq!(stats.frames_relayed, 1);
    }

    // 15. ProtocolBridge passthrough detection
    #[test]
    fn test_protocol_bridge_passthrough() {
        let bridge = ProtocolBridge::new(RelayProtocol::Rtmp, RelayProtocol::Rtmp);
        assert!(bridge.is_passthrough());
    }

    // 16. ProtocolBridge transcode required
    #[test]
    fn test_protocol_bridge_transcode() {
        let bridge = ProtocolBridge::new(RelayProtocol::Rtmp, RelayProtocol::Hls);
        assert!(bridge.requires_transcode);
    }

    // 17. StreamRelay push and process
    #[test]
    fn test_stream_relay_push_process() {
        let mut relay = StreamRelay::new(RelayProtocol::Srt, RelayProtocol::Rtmp, 10);
        relay.push_input(RelayFrame {
            seq: 0,
            timestamp_ms: 0,
            data: vec![0u8; 188],
            is_keyframe: true,
        });
        relay.process();
        assert_eq!(relay.frames_relayed(), 1);
        assert!(relay.pop_output().is_some());
    }

    // 18. StreamRelay drops when queue full
    #[test]
    fn test_stream_relay_drop_when_full() {
        let mut relay = StreamRelay::new(RelayProtocol::Srt, RelayProtocol::Rtmp, 1);
        for i in 0..5u64 {
            relay.push_input(RelayFrame {
                seq: i,
                timestamp_ms: i * 33,
                data: vec![0u8; 4],
                is_keyframe: false,
            });
        }
        // Queue depth is 1, so 4 should be dropped
        assert!(relay.frames_dropped() > 0);
    }

    // 19. StreamRelay protocol bridge
    #[test]
    fn test_stream_relay_protocol_bridge() {
        let relay = StreamRelay::new(RelayProtocol::Srt, RelayProtocol::Rtmp, 10);
        let bridge = relay.protocol_bridge();
        assert_eq!(bridge.from, RelayProtocol::Srt);
        assert_eq!(bridge.to, RelayProtocol::Rtmp);
    }

    // 20. Endpoint stream key and passphrase
    #[test]
    fn test_endpoint_key_and_passphrase() {
        let ep = make_srt_endpoint();
        assert_eq!(ep.passphrase.as_deref(), Some("secret"));
        let ep2 = make_rtmp_endpoint();
        assert!(ep2.stream_key.is_some());
    }
}
