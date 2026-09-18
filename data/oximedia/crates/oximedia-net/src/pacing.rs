//! Packet pacing for SRT and other UDP-based protocols.
//!
//! Packet pacing smooths out bursty traffic by spacing packet transmissions
//! evenly over time, reducing jitter and network congestion.
//!
//! Without pacing, a sender can burst all packets for a video frame
//! simultaneously, causing transient queue overflows in network equipment.
//! Pacing schedules each packet `T/N` seconds apart where `T` is the packet's
//! presentation interval and `N` is the number of packets per frame.
//!
//! Key types:
//! - [`PacingConfig`] — configuration (target bitrate, burst factor, etc.)
//! - [`PacedPacket`] — a packet queued for paced transmission.
//! - [`PacketPacer`] — the pacer: accepts packets and gates their release.
//! - [`PacingStats`] — telemetry: queue depth, inter-packet gap, utilisation.
//!
//! The pacer uses a token-bucket model:
//! - Tokens accumulate at the target bitrate.
//! - Each packet consumes `packet_bytes × 8` tokens.
//! - When the bucket has enough tokens the packet is released.
//! - A configurable burst factor limits how many tokens can accumulate.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Packet pacer configuration.
#[derive(Debug, Clone)]
pub struct PacingConfig {
    /// Target transmission bitrate in bits per second.
    pub target_bitrate_bps: f64,
    /// Burst factor: max tokens = burst_factor × target_bitrate_bps × 1 s.
    /// Typical values: 1.0 (no burst) – 2.0 (allow 2× burst).
    pub burst_factor: f64,
    /// Maximum number of packets to queue.
    pub max_queue_depth: usize,
    /// Whether to prioritise keyframe/control packets over regular data.
    pub priority_queue: bool,
    /// Minimum inter-packet gap (floor to prevent starvation).
    pub min_gap: Duration,
    /// Maximum inter-packet gap (ceiling to avoid indefinite hold).
    pub max_gap: Duration,
}

impl Default for PacingConfig {
    fn default() -> Self {
        Self {
            target_bitrate_bps: 4_000_000.0, // 4 Mbps default
            burst_factor: 1.5,
            max_queue_depth: 512,
            priority_queue: true,
            min_gap: Duration::from_micros(100),
            max_gap: Duration::from_millis(50),
        }
    }
}

impl PacingConfig {
    /// Creates a new configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the target bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, bps: f64) -> Self {
        self.target_bitrate_bps = bps;
        self
    }

    /// Sets the burst factor.
    #[must_use]
    pub fn with_burst_factor(mut self, factor: f64) -> Self {
        self.burst_factor = factor.clamp(1.0, 10.0);
        self
    }

    /// Computes the ideal inter-packet gap for a packet of `packet_bytes` bytes.
    #[must_use]
    pub fn ideal_gap(&self, packet_bytes: usize) -> Duration {
        if self.target_bitrate_bps <= 0.0 {
            return self.max_gap;
        }
        let bits = (packet_bytes as f64) * 8.0;
        let secs = bits / self.target_bitrate_bps;
        let gap = Duration::from_secs_f64(secs);
        gap.clamp(self.min_gap, self.max_gap)
    }

    /// Returns the maximum token bucket capacity in bits.
    #[must_use]
    pub fn max_bucket_bits(&self) -> f64 {
        self.target_bitrate_bps * self.burst_factor
    }
}

// ─── Priority ─────────────────────────────────────────────────────────────────

/// Transmission priority for paced packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PacketPriority {
    /// Control / retransmission (highest priority).
    Control = 3,
    /// Video keyframe or audio.
    High = 2,
    /// Regular data packet.
    Normal = 1,
    /// Background / filler.
    Low = 0,
}

// ─── Paced Packet ─────────────────────────────────────────────────────────────

/// A packet queued in the pacer.
#[derive(Debug)]
pub struct PacedPacket {
    /// Packet payload.
    pub payload: Vec<u8>,
    /// Transmission priority.
    pub priority: PacketPriority,
    /// Earliest allowed send time (pacing deadline).
    pub not_before: Instant,
    /// When this packet was enqueued.
    pub enqueued_at: Instant,
    /// RTP sequence number (for SRT/RTP context).
    pub seq: u32,
}

impl PacedPacket {
    /// Creates a new paced packet with `Normal` priority.
    #[must_use]
    pub fn new(payload: Vec<u8>, seq: u32, not_before: Instant) -> Self {
        Self {
            payload,
            priority: PacketPriority::Normal,
            not_before,
            enqueued_at: Instant::now(),
            seq,
        }
    }

    /// Sets the priority.
    #[must_use]
    pub const fn with_priority(mut self, priority: PacketPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Returns the packet size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.payload.len()
    }

    /// Returns the queuing delay so far.
    #[must_use]
    pub fn queue_latency(&self) -> Duration {
        self.enqueued_at.elapsed()
    }
}

// ─── Token Bucket ─────────────────────────────────────────────────────────────

/// A token-bucket rate limiter.
struct TokenBucket {
    /// Current tokens (bits).
    tokens: f64,
    /// Maximum tokens (bits).
    max_tokens: f64,
    /// Token fill rate (bits per second).
    fill_rate_bps: f64,
    /// Last update time.
    last_fill: Instant,
}

impl TokenBucket {
    fn new(fill_rate_bps: f64, max_tokens: f64) -> Self {
        Self {
            tokens: max_tokens, // Start full.
            max_tokens,
            fill_rate_bps,
            last_fill: Instant::now(),
        }
    }

    /// Refills the bucket based on elapsed time.
    fn refill(&mut self) {
        let elapsed = self.last_fill.elapsed().as_secs_f64();
        let new_tokens = elapsed * self.fill_rate_bps;
        self.tokens = (self.tokens + new_tokens).min(self.max_tokens);
        self.last_fill = Instant::now();
    }

    /// Tries to consume `bits` tokens.  Returns `true` if successful.
    fn try_consume(&mut self, bits: f64) -> bool {
        self.refill();
        if self.tokens >= bits {
            self.tokens -= bits;
            true
        } else {
            false
        }
    }

    /// Returns the current fill level as a fraction [0.0, 1.0].
    fn fill_fraction(&self) -> f64 {
        if self.max_tokens <= 0.0 {
            return 0.0;
        }
        (self.tokens / self.max_tokens).clamp(0.0, 1.0)
    }

    /// Updates the fill rate (on bitrate change).
    fn set_rate(&mut self, fill_rate_bps: f64, max_tokens: f64) {
        self.fill_rate_bps = fill_rate_bps;
        self.max_tokens = max_tokens;
        self.tokens = self.tokens.min(max_tokens);
    }
}

// ─── Pacer Statistics ─────────────────────────────────────────────────────────

/// Packet pacer telemetry.
#[derive(Debug, Clone, Default)]
pub struct PacingStats {
    /// Current queue depth.
    pub queue_depth: usize,
    /// Total packets sent.
    pub packets_sent: u64,
    /// Total packets dropped (queue overflow).
    pub packets_dropped: u64,
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Average queuing latency of sent packets.
    pub avg_queue_latency: Duration,
    /// Current estimated output bitrate.
    pub output_bitrate_bps: f64,
    /// Token bucket fill fraction (0.0 – 1.0).
    pub bucket_fill: f64,
}

// ─── Packet Pacer ─────────────────────────────────────────────────────────────

/// Token-bucket packet pacer.
///
/// Enqueue packets via `enqueue`; call `dequeue_ready` on each
/// scheduling tick to obtain packets that are cleared to transmit.
pub struct PacketPacer {
    /// Configuration.
    config: PacingConfig,
    /// High-priority queue (control / keyframes).
    high_queue: VecDeque<PacedPacket>,
    /// Normal-priority queue.
    normal_queue: VecDeque<PacedPacket>,
    /// Low-priority queue.
    low_queue: VecDeque<PacedPacket>,
    /// Token bucket.
    bucket: TokenBucket,
    /// Next allowed send time (hard inter-packet gap floor).
    next_send: Instant,
    /// Statistics.
    stats: PacingStats,
    /// Smoothed throughput estimator.
    throughput_ema: f64,
    /// Last throughput update time.
    last_throughput_update: Instant,
    /// Bytes in the current measurement window.
    window_bytes: u64,
    /// Window start time.
    window_start: Instant,
}

impl PacketPacer {
    /// Creates a new packet pacer.
    #[must_use]
    pub fn new(config: PacingConfig) -> Self {
        let max_bits = config.max_bucket_bits();
        let rate = config.target_bitrate_bps;
        let bucket = TokenBucket::new(rate, max_bits);
        let now = Instant::now();
        let mut pacer = Self {
            config,
            high_queue: VecDeque::new(),
            normal_queue: VecDeque::new(),
            low_queue: VecDeque::new(),
            bucket,
            next_send: now,
            stats: PacingStats::default(),
            throughput_ema: 0.0,
            last_throughput_update: now,
            window_bytes: 0,
            window_start: now,
        };
        // Populate initial stats so callers see bucket_fill = 1.0 immediately.
        pacer.update_stats();
        pacer
    }

    /// Enqueues a packet for paced transmission.
    ///
    /// Returns `true` if accepted, `false` if the queue is full (packet dropped).
    pub fn enqueue(&mut self, packet: PacedPacket) -> bool {
        let total_depth = self.high_queue.len() + self.normal_queue.len() + self.low_queue.len();
        if total_depth >= self.config.max_queue_depth {
            self.stats.packets_dropped += 1;
            return false;
        }

        match packet.priority {
            PacketPriority::Control | PacketPriority::High => {
                self.high_queue.push_back(packet);
            }
            PacketPriority::Normal => {
                self.normal_queue.push_back(packet);
            }
            PacketPriority::Low => {
                self.low_queue.push_back(packet);
            }
        }

        self.update_stats();
        true
    }

    /// Dequeues packets that are ready to send right now.
    ///
    /// Packets are selected in priority order.  Each packet is checked against
    /// the token bucket; if tokens are insufficient the dequeue stops (to avoid
    /// unbounded bursting).
    ///
    /// Call this method at a rate higher than your target bitrate to ensure
    /// smooth pacing (e.g. every millisecond for multi-Mbps streams).
    pub fn dequeue_ready(&mut self) -> Vec<PacedPacket> {
        let mut out = Vec::new();

        loop {
            // Refresh `now` each iteration so the inter-packet gap floor is
            // evaluated against the actual current time, not a stale snapshot.
            let now = Instant::now();

            // Pick the next packet from the highest-priority non-empty queue.
            let next = if let Some(p) = self.high_queue.front() {
                if p.not_before <= now {
                    self.high_queue.pop_front()
                } else {
                    None
                }
            } else if let Some(p) = self.normal_queue.front() {
                if p.not_before <= now {
                    self.normal_queue.pop_front()
                } else {
                    None
                }
            } else if let Some(p) = self.low_queue.front() {
                if p.not_before <= now {
                    self.low_queue.pop_front()
                } else {
                    None
                }
            } else {
                break;
            };

            let packet = match next {
                Some(p) => p,
                None => break,
            };

            // Check inter-packet gap floor.
            if now < self.next_send {
                // Re-insert at front of appropriate queue.
                match packet.priority {
                    PacketPriority::Control | PacketPriority::High => {
                        self.high_queue.push_front(packet);
                    }
                    PacketPriority::Normal => {
                        self.normal_queue.push_front(packet);
                    }
                    PacketPriority::Low => {
                        self.low_queue.push_front(packet);
                    }
                }
                break;
            }

            // Token bucket check.
            let bits = packet.size_bytes() as f64 * 8.0;
            if !self.bucket.try_consume(bits) {
                // Re-insert and stop draining.
                match packet.priority {
                    PacketPriority::Control | PacketPriority::High => {
                        self.high_queue.push_front(packet);
                    }
                    PacketPriority::Normal => {
                        self.normal_queue.push_front(packet);
                    }
                    PacketPriority::Low => {
                        self.low_queue.push_front(packet);
                    }
                }
                break;
            }

            // Update next_send gap floor.
            // Use the bitrate-computed gap, clamped to [0, max_gap].
            // min_gap is applied as a floor only when the ideal gap exceeds it,
            // so that very-high-bitrate streams (where ideal gap << min_gap)
            // are not artificially serialised at the min_gap rate.
            let raw_gap = if self.config.target_bitrate_bps > 0.0 {
                let bits_f = (packet.size_bytes() as f64) * 8.0;
                Duration::from_secs_f64(bits_f / self.config.target_bitrate_bps)
            } else {
                self.config.max_gap
            };
            let gap = if raw_gap >= self.config.min_gap {
                raw_gap.min(self.config.max_gap)
            } else {
                // Raw gap is smaller than min_gap floor — don't enforce any gap,
                // letting the token bucket alone control the rate.
                Duration::ZERO
            };
            self.next_send = now + gap;

            // Update throughput measurement.
            self.window_bytes += packet.size_bytes() as u64;
            self.stats.bytes_sent += packet.size_bytes() as u64;
            self.stats.packets_sent += 1;

            out.push(packet);
        }

        self.update_throughput();
        self.update_stats();
        out
    }

    /// Updates the target bitrate at runtime.
    pub fn set_bitrate(&mut self, bps: f64) {
        self.config.target_bitrate_bps = bps;
        let max_bits = self.config.max_bucket_bits();
        self.bucket.set_rate(bps, max_bits);
    }

    /// Returns the current configuration.
    #[must_use]
    pub const fn config(&self) -> &PacingConfig {
        &self.config
    }

    /// Returns a snapshot of the current statistics.
    #[must_use]
    pub fn stats(&self) -> &PacingStats {
        &self.stats
    }

    /// Returns the total queue depth.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.high_queue.len() + self.normal_queue.len() + self.low_queue.len()
    }

    /// Returns the number of high-priority packets queued.
    #[must_use]
    pub fn high_priority_depth(&self) -> usize {
        self.high_queue.len()
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn update_stats(&mut self) {
        self.stats.queue_depth = self.queue_depth();
        self.stats.bucket_fill = self.bucket.fill_fraction();
    }

    fn update_throughput(&mut self) {
        let elapsed = self.window_start.elapsed();
        if elapsed >= Duration::from_millis(100) {
            let bps = (self.window_bytes as f64 * 8.0) / elapsed.as_secs_f64();
            self.throughput_ema = 0.8 * self.throughput_ema + 0.2 * bps;
            self.stats.output_bitrate_bps = self.throughput_ema;
            self.window_bytes = 0;
            self.window_start = Instant::now();
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_packet(seq: u32, size: usize) -> PacedPacket {
        let not_before = Instant::now(); // Immediately eligible.
        PacedPacket::new(vec![0u8; size], seq, not_before)
    }

    fn make_high_packet(seq: u32, size: usize) -> PacedPacket {
        make_packet(seq, size).with_priority(PacketPriority::Control)
    }

    // 1. Default config
    #[test]
    fn test_pacing_config_default() {
        let cfg = PacingConfig::default();
        assert!(cfg.target_bitrate_bps > 0.0);
        assert!(cfg.burst_factor >= 1.0);
        assert!(cfg.max_queue_depth > 0);
    }

    // 2. Config builder
    #[test]
    fn test_pacing_config_builder() {
        let cfg = PacingConfig::new()
            .with_bitrate(10_000_000.0)
            .with_burst_factor(2.0);
        assert!((cfg.target_bitrate_bps - 10_000_000.0).abs() < 1.0);
        assert!((cfg.burst_factor - 2.0).abs() < 1e-9);
    }

    // 3. Ideal gap calculation
    #[test]
    fn test_ideal_gap() {
        let cfg = PacingConfig::new().with_bitrate(1_000_000.0); // 1 Mbps
                                                                 // 1250 bytes = 10000 bits → gap = 10 ms at 1 Mbps
        let gap = cfg.ideal_gap(1250);
        assert!(gap >= Duration::from_millis(9) && gap <= Duration::from_millis(11));
    }

    // 4. Max bucket bits
    #[test]
    fn test_max_bucket_bits() {
        let cfg = PacingConfig::new()
            .with_bitrate(1_000_000.0)
            .with_burst_factor(2.0);
        assert!((cfg.max_bucket_bits() - 2_000_000.0).abs() < 1.0);
    }

    // 5. PacedPacket size
    #[test]
    fn test_paced_packet_size() {
        let pkt = make_packet(0, 188);
        assert_eq!(pkt.size_bytes(), 188);
    }

    // 6. PacedPacket priority
    #[test]
    fn test_paced_packet_priority() {
        let pkt = make_packet(0, 100).with_priority(PacketPriority::High);
        assert_eq!(pkt.priority, PacketPriority::High);
    }

    // 7. Enqueue and dequeue
    #[test]
    fn test_pacer_enqueue_dequeue() {
        // Use high bitrate so token bucket never throttles in this test.
        let cfg = PacingConfig::new().with_bitrate(1_000_000_000.0); // 1 Gbps
        let mut pacer = PacketPacer::new(cfg);
        assert!(pacer.enqueue(make_packet(0, 188)));
        let out = pacer.dequeue_ready();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].seq, 0);
    }

    // 8. Queue depth tracking
    #[test]
    fn test_pacer_queue_depth() {
        let cfg = PacingConfig::new().with_bitrate(1.0); // Very slow so nothing dequeues
        let mut pacer = PacketPacer::new(cfg);
        pacer.enqueue(make_packet(0, 1));
        pacer.enqueue(make_packet(1, 1));
        assert_eq!(pacer.queue_depth(), 2);
    }

    // 9. Queue overflow drops packet
    #[test]
    fn test_pacer_queue_overflow() {
        let mut cfg = PacingConfig::new();
        cfg.max_queue_depth = 2;
        cfg.target_bitrate_bps = 1.0; // Slow drain
        let mut pacer = PacketPacer::new(cfg);
        pacer.enqueue(make_packet(0, 1));
        pacer.enqueue(make_packet(1, 1));
        let dropped = !pacer.enqueue(make_packet(2, 1)); // Should be dropped
        assert!(dropped);
        assert_eq!(pacer.stats().packets_dropped, 1);
    }

    // 10. High priority packet dequeues first
    #[test]
    fn test_pacer_priority_order() {
        let cfg = PacingConfig::new().with_bitrate(1_000_000_000.0);
        let mut pacer = PacketPacer::new(cfg);
        pacer.enqueue(make_packet(1, 10).with_priority(PacketPriority::Low));
        pacer.enqueue(make_high_packet(2, 10));
        let out = pacer.dequeue_ready();
        assert_eq!(out.len(), 2);
        // High-priority (seq=2) should come first.
        assert_eq!(out[0].priority, PacketPriority::Control);
    }

    // 11. Set bitrate at runtime
    #[test]
    fn test_pacer_set_bitrate() {
        let mut pacer = PacketPacer::new(PacingConfig::default());
        pacer.set_bitrate(10_000_000.0);
        assert!((pacer.config().target_bitrate_bps - 10_000_000.0).abs() < 1.0);
    }

    // 12. Stats update after dequeue
    #[test]
    fn test_pacer_stats_after_dequeue() {
        let cfg = PacingConfig::new().with_bitrate(1_000_000_000.0);
        let mut pacer = PacketPacer::new(cfg);
        pacer.enqueue(make_packet(0, 200));
        pacer.dequeue_ready();
        assert_eq!(pacer.stats().packets_sent, 1);
        assert_eq!(pacer.stats().bytes_sent, 200);
    }

    // 13. High-priority queue depth
    #[test]
    fn test_pacer_high_priority_depth() {
        let cfg = PacingConfig::new().with_bitrate(1.0); // Slow drain
        let mut pacer = PacketPacer::new(cfg);
        pacer.enqueue(make_high_packet(0, 1));
        pacer.enqueue(make_high_packet(1, 1));
        assert_eq!(pacer.high_priority_depth(), 2);
    }

    // 14. Packet queue latency is non-negative
    #[test]
    fn test_paced_packet_queue_latency() {
        let pkt = make_packet(0, 100);
        let latency = pkt.queue_latency();
        assert!(latency >= Duration::ZERO);
    }

    // 15. PacketPriority ordering
    #[test]
    fn test_packet_priority_ordering() {
        assert!(PacketPriority::Control > PacketPriority::High);
        assert!(PacketPriority::High > PacketPriority::Normal);
        assert!(PacketPriority::Normal > PacketPriority::Low);
    }

    // 16. Token bucket fill starts at max
    #[test]
    fn test_token_bucket_starts_full() {
        let cfg = PacingConfig::new().with_bitrate(1_000_000.0);
        let pacer = PacketPacer::new(cfg);
        // Bucket starts full, so fill fraction should be 1.0
        assert!((pacer.stats().bucket_fill - 1.0).abs() < 1e-3);
    }

    // 17. Zero-length payload
    #[test]
    fn test_pacer_zero_length_payload() {
        let cfg = PacingConfig::new().with_bitrate(1_000_000_000.0);
        let mut pacer = PacketPacer::new(cfg);
        pacer.enqueue(make_packet(0, 0));
        let out = pacer.dequeue_ready();
        assert_eq!(out.len(), 1);
    }

    // 18. Multiple enqueues and dequeues
    #[test]
    fn test_pacer_multiple_packets() {
        let cfg = PacingConfig::new().with_bitrate(1_000_000_000.0);
        let mut pacer = PacketPacer::new(cfg);
        for i in 0..10u32 {
            pacer.enqueue(make_packet(i, 100));
        }
        let out = pacer.dequeue_ready();
        // Should get all 10 at 1 Gbps
        assert_eq!(out.len(), 10);
    }

    // 19. Ideal gap zero bitrate returns max_gap
    #[test]
    fn test_ideal_gap_zero_bitrate() {
        let cfg = PacingConfig::new().with_bitrate(0.0);
        let gap = cfg.ideal_gap(1000);
        assert_eq!(gap, cfg.max_gap);
    }

    // 20. Enqueue into all priority queues
    #[test]
    fn test_pacer_all_priority_queues() {
        let cfg = PacingConfig::new().with_bitrate(1.0); // Slow
        let mut pacer = PacketPacer::new(cfg);
        pacer.enqueue(make_packet(0, 1).with_priority(PacketPriority::Low));
        pacer.enqueue(make_packet(1, 1).with_priority(PacketPriority::Normal));
        pacer.enqueue(make_packet(2, 1).with_priority(PacketPriority::High));
        pacer.enqueue(make_packet(3, 1).with_priority(PacketPriority::Control));
        assert_eq!(pacer.queue_depth(), 4);
        assert_eq!(pacer.high_priority_depth(), 2); // Control + High
    }
}
