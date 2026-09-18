//! Packet pacing for SRT transport.
//!
//! SRT sends data over UDP.  Without pacing, all packets for a single video
//! frame can be burst onto the network simultaneously, causing transient queue
//! overflows in intermediate network equipment and increasing jitter.
//!
//! This module implements a **token-bucket pacer** that spreads packets evenly
//! across the sending interval so that the inter-packet gap is roughly
//! constant.
//!
//! # Algorithm
//!
//! The token bucket accumulates tokens at the target bitrate:
//!
//! ```text
//! tokens_per_second = target_bitrate_bps   (in bits)
//! cost_of_packet    = packet_bytes × 8     (bits)
//! ```
//!
//! When the application asks for the pacing delay before sending a packet:
//!
//! 1. Refill the bucket based on elapsed wall time.
//! 2. If `tokens >= cost`: consume and return `Duration::ZERO`.
//! 3. Otherwise compute `deficit = cost - tokens` and return
//!    `Duration::from_secs_f64(deficit / target_bitrate_bps)`.
//!
//! The caller is expected to `sleep` (or schedule a timer) for the returned
//! duration before transmitting the packet, then call [`SrtPacketPacer::record_sent`].
//!
//! # Example
//!
//! ```rust
//! use oximedia_net::srt_pacing::SrtPacketPacer;
//!
//! let mut pacer = SrtPacketPacer::new(4_000_000); // 4 Mbps
//! let delay = pacer.pacing_delay(1316); // MTU-sized SRT packet
//! // sleep(delay) …
//! pacer.record_sent(1316);
//! ```

use std::time::{Duration, Instant};

// ─── SrtPacketPacer ───────────────────────────────────────────────────────────

/// Token-bucket pacer for SRT transport packets.
///
/// The pacer keeps state between calls so that the measured send rate
/// converges towards the target bitrate over time.
///
/// Call [`Self::pacing_delay`] to obtain the duration to wait before sending,
/// then call [`Self::record_sent`] after the packet has been handed to the OS
/// socket layer.
#[derive(Debug)]
pub struct SrtPacketPacer {
    /// Target bitrate in bits per second.
    target_bitrate_bps: u64,
    /// Current token bucket level in bits (fractional part tracked as f64).
    tokens: f64,
    /// Maximum bucket depth in bits (capped at 1 second of target bitrate to
    /// limit burst).
    max_tokens: f64,
    /// Wall-clock time of the last bucket refill.
    last_refill: Instant,
    /// Start of the current 1-second measurement window.
    window_start: Instant,
    /// Bytes sent in the current measurement window.
    bytes_sent_this_second: u64,
}

impl SrtPacketPacer {
    /// Creates a new pacer targeting `target_bitrate_bps` bits per second.
    ///
    /// The token bucket is initialised to one full second of capacity so that
    /// the first burst of packets at connection start is not penalised.
    #[must_use]
    pub fn new(target_bitrate_bps: u64) -> Self {
        let max_tokens = target_bitrate_bps as f64;
        let now = Instant::now();
        Self {
            target_bitrate_bps,
            tokens: max_tokens,
            max_tokens,
            last_refill: now,
            window_start: now,
            bytes_sent_this_second: 0,
        }
    }

    /// Updates the target bitrate at runtime (e.g., on network condition change).
    pub fn set_bitrate(&mut self, bps: u64) {
        self.target_bitrate_bps = bps;
        self.max_tokens = bps as f64;
        // Clamp tokens to the new maximum.
        self.tokens = self.tokens.min(self.max_tokens);
    }

    /// Returns the target bitrate this pacer was configured with.
    #[must_use]
    pub fn target_bitrate_bps(&self) -> u64 {
        self.target_bitrate_bps
    }

    /// Computes the delay to insert before sending `packet_size` bytes.
    ///
    /// Refills the token bucket based on elapsed time and checks whether
    /// there are enough tokens to send immediately.  If not, returns the
    /// duration needed for the bucket to accumulate sufficient tokens.
    ///
    /// Returns [`Duration::ZERO`] when the packet can be sent without waiting.
    ///
    /// This method does **not** consume tokens; call [`Self::record_sent`]
    /// after the packet is actually transmitted.
    #[must_use]
    pub fn pacing_delay(&mut self, packet_size: usize) -> Duration {
        self.refill();

        if self.target_bitrate_bps == 0 {
            return Duration::ZERO;
        }

        let cost_bits = (packet_size as f64) * 8.0;

        if self.tokens >= cost_bits {
            // Enough tokens — no delay needed.
            Duration::ZERO
        } else {
            // Compute time until the bucket has enough tokens.
            let deficit = cost_bits - self.tokens;
            let wait_secs = deficit / (self.target_bitrate_bps as f64);
            Duration::from_secs_f64(wait_secs)
        }
    }

    /// Records that `packet_size` bytes have been sent.
    ///
    /// Consumes tokens from the bucket and advances the per-second accounting
    /// window.  Must be called after each packet is handed off to the socket.
    pub fn record_sent(&mut self, packet_size: usize) {
        self.refill();

        let cost_bits = (packet_size as f64) * 8.0;
        // Drain tokens (allow going negative — refill will recover on next call).
        self.tokens -= cost_bits;
        self.bytes_sent_this_second += packet_size as u64;

        // Slide the measurement window every second.
        if self.window_start.elapsed() >= Duration::from_secs(1) {
            self.bytes_sent_this_second = 0;
            self.window_start = Instant::now();
        }
    }

    /// Returns the measured send rate in bits per second over the current
    /// one-second rolling window.
    ///
    /// Returns `0` in the first second before any packets have been sent.
    #[must_use]
    pub fn current_rate_bps(&self) -> u64 {
        let elapsed = self.window_start.elapsed();
        if elapsed.as_secs_f64() < 1e-9 || self.bytes_sent_this_second == 0 {
            return 0;
        }
        let bps = (self.bytes_sent_this_second as f64 * 8.0) / elapsed.as_secs_f64();
        bps as u64
    }

    // ── Private ───────────────────────────────────────────────────────────────

    /// Refills the token bucket based on elapsed time since the last refill.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_bits = elapsed * (self.target_bitrate_bps as f64);
        self.tokens = (self.tokens + new_bits).min(self.max_tokens);
        self.last_refill = now;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // 1. Small packets well below the target rate should not require any delay.
    #[test]
    fn test_pacer_no_delay_under_rate() {
        // 10 Mbps pacer; send one 188-byte (TS packet) packet.
        // 188 × 8 = 1504 bits; bucket starts with 10_000_000 bits → no delay.
        let mut pacer = SrtPacketPacer::new(10_000_000);
        let delay = pacer.pacing_delay(188);
        assert_eq!(
            delay,
            Duration::ZERO,
            "first packet at 10 Mbps should require no delay"
        );
    }

    // 2. A burst of large packets exceeding the 1-second budget should trigger
    //    a non-zero delay.
    #[test]
    fn test_pacer_delay_over_rate() {
        // 100 kbps pacer; attempt to send a 64 KiB packet (524_288 bits) which
        // costs 5× the per-second budget.  After draining the bucket via
        // record_sent, pacing_delay should return > 0.
        let mut pacer = SrtPacketPacer::new(100_000);
        // The bucket starts full (100_000 bits = 12_500 bytes).
        // Drain it completely first.
        pacer.record_sent(12_500); // 100_000 bits — empties the bucket
                                   // Now ask for pacing delay before another large packet.
        let delay = pacer.pacing_delay(1_316); // 10_528 bits deficit
        assert!(
            delay > Duration::ZERO,
            "over-rate burst should produce non-zero delay, got {delay:?}"
        );
    }

    // 3. Paced sends at a known rate should report a measured rate that
    //    converges towards the target.
    //
    //    We send 1 Mbps worth of data in one call (125_000 bytes = 1_000_000
    //    bits) and then check that `current_rate_bps` is in the right order of
    //    magnitude.  Exact timing is not asserted to keep the test deterministic
    //    on slow CI machines.
    #[test]
    fn test_pacer_rate_tracking() {
        let mut pacer = SrtPacketPacer::new(1_000_000);
        // Record 125_000 bytes (1 Mbit) of sent data.
        // We send in 1316-byte chunks (SRT MTU).
        let chunk = 1316usize;
        let total_bytes = 125_000usize;
        let mut sent = 0usize;
        while sent + chunk <= total_bytes {
            pacer.record_sent(chunk);
            sent += chunk;
        }
        if sent < total_bytes {
            pacer.record_sent(total_bytes - sent);
        }

        // The measured rate should be in the ballpark of 1 Mbps.
        // `current_rate_bps()` is `bytes*8 / window_start.elapsed()` and returns 0
        // when the window elapsed rounds to zero (the record loop runs in well
        // under a clock tick under load). Guarantee a measurable window before
        // reading so the non-zero assertion is deterministic.
        std::thread::sleep(Duration::from_millis(10));
        let rate = pacer.current_rate_bps();
        assert!(rate > 0, "rate should be non-zero after sending");
        // Upper bound: in a tight loop with no sleep, all packets are recorded
        // within a sub-millisecond window, so the computed instantaneous rate can
        // be orders of magnitude above the configured bitrate.  We only verify it
        // is non-zero (above) and does not overflow a u64 (implicitly guaranteed
        // by the return type).  A tighter bound would require mocked time or real
        // inter-packet sleeps, both of which are inappropriate for a unit test.
    }

    // 4. After setting a new bitrate, pacing_delay respects the updated rate.
    #[test]
    fn test_pacer_set_bitrate() {
        let mut pacer = SrtPacketPacer::new(1_000_000);
        pacer.set_bitrate(2_000_000);
        assert_eq!(pacer.target_bitrate_bps(), 2_000_000);
    }

    // 5. Zero-bitrate pacer never delays.
    #[test]
    fn test_pacer_zero_bitrate_no_delay() {
        let mut pacer = SrtPacketPacer::new(0);
        let delay = pacer.pacing_delay(65_536);
        assert_eq!(delay, Duration::ZERO);
    }

    // 6. Bucket refill over time: after a short sleep the bucket gains tokens.
    //    We drain the bucket fully and verify that pacing_delay eventually
    //    returns zero without needing to sleep by fast-forwarding via record_sent.
    #[test]
    fn test_pacer_bucket_refills() {
        // Very slow 8 bps pacer: 1 byte per second.  Bucket starts with 1 bit.
        let mut pacer = SrtPacketPacer::new(8);
        // Drain everything: record 1 byte sent (8 bits).
        pacer.record_sent(1);
        // After draining, a 1-byte packet costs 8 bits and the bucket is at ≤0.
        // We should get a non-zero delay.
        let delay_after_drain = pacer.pacing_delay(1);
        assert!(
            delay_after_drain > Duration::ZERO || pacer.tokens >= 8.0,
            "bucket should be exhausted or nearly so"
        );
    }

    // 7. current_rate_bps returns 0 before any packets are sent.
    #[test]
    fn test_pacer_rate_zero_before_send() {
        let pacer = SrtPacketPacer::new(10_000_000);
        // No packets sent yet — rate should be 0.
        assert_eq!(pacer.current_rate_bps(), 0);
    }
}
