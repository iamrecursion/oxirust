//! RTMP chunk-size optimisation and session statistics.
//!
//! The RTMP specification (§5.4.1) allows either peer to change the chunk size
//! at any time via a *Set Chunk Size* control message.  Choosing a good chunk
//! size balances:
//!
//! - **Multiplexing latency** — smaller chunks let audio interleave more
//!   frequently with large video chunks.
//! - **Per-chunk overhead** — larger chunks amortise the 7-byte header over
//!   more payload bytes.
//!
//! [`RtmpChunkOptimizer`] provides an adaptive strategy: it computes an
//! analytically optimal chunk size from the available bandwidth and then
//! adjusts it in response to observed packet loss.
//!
//! [`RtmpSessionStats`] tracks bytes sent, chunk counts, and dropped frames for
//! monitoring dashboards and adaptive control loops.

// ─── Session Statistics ───────────────────────────────────────────────────────

/// Running statistics for an RTMP session.
///
/// The fields are updated by the optimizer's [`RtmpChunkOptimizer::record_sent`]
/// helper, but callers may also update them directly when integrating with a
/// custom chunk layer.
#[derive(Debug, Clone)]
pub struct RtmpSessionStats {
    /// Cumulative bytes successfully sent (payload only, excluding headers).
    pub total_bytes: u64,
    /// Number of chunks sent since the last [`RtmpSessionStats::reset`].
    pub chunks_sent: u64,
    /// Exponentially-smoothed average chunk size (bytes).
    pub avg_chunk_size: f64,
    /// Number of frames dropped due to congestion or buffer overflow.
    pub dropped_frames: u32,
}

impl Default for RtmpSessionStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RtmpSessionStats {
    /// Create zeroed statistics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_bytes: 0,
            chunks_sent: 0,
            avg_chunk_size: 0.0,
            dropped_frames: 0,
        }
    }

    /// Record a successfully sent chunk of `chunk_size` bytes.
    ///
    /// Updates `total_bytes`, `chunks_sent`, and `avg_chunk_size` using a
    /// cumulative moving average so that `avg_chunk_size` is always
    /// `total_bytes / chunks_sent`.
    pub fn record_chunk(&mut self, chunk_size: u32) {
        self.total_bytes += u64::from(chunk_size);
        self.chunks_sent += 1;
        // Cumulative moving average: avoid division by zero (chunks_sent >= 1).
        self.avg_chunk_size = self.total_bytes as f64 / self.chunks_sent as f64;
    }

    /// Increment the dropped-frames counter.
    pub fn drop_frame(&mut self) {
        self.dropped_frames += 1;
    }

    /// Reset all statistics to zero.
    pub fn reset(&mut self) {
        self.total_bytes = 0;
        self.chunks_sent = 0;
        self.avg_chunk_size = 0.0;
        self.dropped_frames = 0;
    }

    /// Compute throughput in bits per second given a measurement window.
    ///
    /// Returns `0.0` when `elapsed_secs` is non-positive.
    #[must_use]
    pub fn throughput_bps(&self, elapsed_secs: f64) -> f64 {
        if elapsed_secs <= 0.0 {
            return 0.0;
        }
        (self.total_bytes as f64 * 8.0) / elapsed_secs
    }
}

// ─── Chunk Optimizer ─────────────────────────────────────────────────────────

/// Minimum allowed RTMP chunk size (bytes).
pub const MIN_CHUNK_SIZE: u32 = 128;
/// Maximum allowed RTMP chunk size (bytes).
pub const MAX_CHUNK_SIZE: u32 = 65536;
/// Sensible default chunk size when no bandwidth estimate is available (bytes).
pub const DEFAULT_CHUNK_SIZE: u32 = 4096;

/// Adaptive RTMP chunk-size controller with integrated session statistics.
///
/// # Algorithm
///
/// The optimal chunk size is derived from the *bandwidth-delay product* heuristic:
///
/// ```text
/// optimal = sqrt(bandwidth_bps / 8 * 0.1)   [bytes]
/// ```
///
/// This targets chunks that consume ≈100 ms of bandwidth at the current rate,
/// which provides a good trade-off between multiplexing granularity and header
/// overhead across typical streaming bandwidths (256 kbps – 100 Mbps).
///
/// The result is then:
/// 1. Rounded up to the nearest multiple of 128 (aligns with RTMP header sizes).
/// 2. Clamped to `[min_chunk_size, max_chunk_size]`.
///
/// Adaptive loss-based adjustment ([`RtmpChunkOptimizer::adaptive_update`])
/// additionally reduces the chunk size by 25 % when packet loss exceeds 5 %,
/// providing back-pressure in congested networks.
#[derive(Debug)]
pub struct RtmpChunkOptimizer {
    /// Lower bound on the negotiated chunk size.
    min_chunk_size: u32,
    /// Upper bound on the negotiated chunk size.
    max_chunk_size: u32,
    /// Currently negotiated chunk size.
    current_chunk_size: u32,
    /// Accumulated session statistics.
    stats: RtmpSessionStats,
}

impl Default for RtmpChunkOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl RtmpChunkOptimizer {
    /// Create an optimizer with default bounds
    /// ([`MIN_CHUNK_SIZE`], [`MAX_CHUNK_SIZE`]) and [`DEFAULT_CHUNK_SIZE`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_chunk_size: MIN_CHUNK_SIZE,
            max_chunk_size: MAX_CHUNK_SIZE,
            current_chunk_size: DEFAULT_CHUNK_SIZE,
            stats: RtmpSessionStats::new(),
        }
    }

    /// Create an optimizer with explicit bounds.
    ///
    /// Both `min` and `max` are clamped to the global
    /// [`MIN_CHUNK_SIZE`]–[`MAX_CHUNK_SIZE`] range.  If `min > max` after
    /// clamping, `min` is set equal to `max`.
    #[must_use]
    pub fn with_bounds(min: u32, max: u32) -> Self {
        let clamped_min = min.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);
        let clamped_max = max.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);
        let (effective_min, effective_max) = if clamped_min <= clamped_max {
            (clamped_min, clamped_max)
        } else {
            (clamped_max, clamped_max)
        };
        Self {
            min_chunk_size: effective_min,
            max_chunk_size: effective_max,
            current_chunk_size: DEFAULT_CHUNK_SIZE.clamp(effective_min, effective_max),
            stats: RtmpSessionStats::new(),
        }
    }

    /// Compute and apply the optimal chunk size for `bandwidth_bps`.
    ///
    /// Stores and returns the new chunk size.
    pub fn optimize_chunk_size(&mut self, bandwidth_bps: u64) -> u32 {
        let raw = ((bandwidth_bps as f64 / 8.0) * 0.1).sqrt();
        // Round up to nearest multiple of 128.
        let rounded = round_to_multiple_of_128(raw as u64);
        let clamped = rounded.clamp(self.min_chunk_size, self.max_chunk_size);
        self.current_chunk_size = clamped;
        clamped
    }

    /// Return the current chunk size without recomputing it.
    #[must_use]
    pub fn current_size(&self) -> u32 {
        self.current_chunk_size
    }

    /// Shared reference to accumulated session statistics.
    #[must_use]
    pub fn stats(&self) -> &RtmpSessionStats {
        &self.stats
    }

    /// Mutable reference to session statistics (for direct manipulation).
    pub fn stats_mut(&mut self) -> &mut RtmpSessionStats {
        &mut self.stats
    }

    /// Record a sent chunk of `bytes` bytes and update statistics.
    pub fn record_sent(&mut self, bytes: u32) {
        self.stats.record_chunk(bytes);
    }

    /// Adaptive chunk-size update that incorporates packet-loss feedback.
    ///
    /// | `packet_loss_pct` | Action |
    /// |---|---|
    /// | > 5.0 | Reduce chunk size by 25 % (clamped to `min_chunk_size`) |
    /// | < 1.0 | Re-optimise for `bandwidth_bps` |
    /// | 1.0 – 5.0 | Keep current size |
    ///
    /// Returns the new current chunk size.
    pub fn adaptive_update(&mut self, bandwidth_bps: u64, packet_loss_pct: f64) -> u32 {
        if packet_loss_pct > 5.0 {
            // Back off by 25%.
            let reduced = (self.current_chunk_size as f64 * 0.75) as u32;
            self.current_chunk_size = reduced.clamp(self.min_chunk_size, self.max_chunk_size);
        } else if packet_loss_pct < 1.0 {
            self.optimize_chunk_size(bandwidth_bps);
        }
        // else: 1.0 <= loss <= 5.0 → hold current size
        self.current_chunk_size
    }

    /// Compute how many chunks of `current_chunk_size` are needed to carry
    /// `total_bytes` bytes (ceiling division).
    ///
    /// Returns `0` when `total_bytes` is zero or `current_chunk_size` is zero.
    #[must_use]
    pub fn suggested_chunk_count(&self, total_bytes: u64) -> u64 {
        if total_bytes == 0 || self.current_chunk_size == 0 {
            return 0;
        }
        let cs = u64::from(self.current_chunk_size);
        (total_bytes + cs - 1) / cs
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Round `value` up to the nearest multiple of 128, returning at least 128.
fn round_to_multiple_of_128(value: u64) -> u32 {
    if value == 0 {
        return 128;
    }
    let rem = value % 128;
    let rounded = if rem == 0 { value } else { value + (128 - rem) };
    // Clamp to u32::MAX to avoid overflow before casting.
    rounded.min(u32::MAX as u64) as u32
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── optimize_chunk_size ──────────────────────────────────────────────────

    #[test]
    fn test_optimize_chunk_size_low_bandwidth_clamped_to_min() {
        // 64 kbps → raw = sqrt(64000/8 * 0.1) = sqrt(800) ≈ 28 → clamped to 128
        let mut opt = RtmpChunkOptimizer::new();
        let size = opt.optimize_chunk_size(64_000);
        assert_eq!(size, MIN_CHUNK_SIZE, "Expected MIN_CHUNK_SIZE at 64 kbps");
    }

    #[test]
    fn test_optimize_chunk_size_medium_bandwidth() {
        // 10 Mbps → raw = sqrt(10_000_000/8 * 0.1) = sqrt(125_000) ≈ 353.6
        // → rounded to 384, clamped within [128, 65536]
        let mut opt = RtmpChunkOptimizer::new();
        let size = opt.optimize_chunk_size(10_000_000);
        assert!(size >= MIN_CHUNK_SIZE, "Must be >= MIN");
        assert!(size <= MAX_CHUNK_SIZE, "Must be <= MAX");
        assert_eq!(size % 128, 0, "Must be a multiple of 128");
        // Verify the formula: expect 384 (353.6 rounded up to nearest 128-multiple)
        assert_eq!(size, 384, "10 Mbps should yield chunk size 384");
    }

    #[test]
    fn test_optimize_chunk_size_high_bandwidth_clamped_to_max() {
        // To hit MAX_CHUNK_SIZE (65536) we need:
        //   sqrt(bw/8 * 0.1) >= 65536  →  bw >= 65536^2 * 80 = ~343 Tbps
        // Use u64::MAX as a proxy for an astronomical bandwidth.
        let mut opt = RtmpChunkOptimizer::new();
        let size = opt.optimize_chunk_size(u64::MAX);
        assert_eq!(
            size, MAX_CHUNK_SIZE,
            "Expected MAX_CHUNK_SIZE for extreme bandwidth"
        );
    }

    #[test]
    fn test_optimize_chunk_size_zero_bandwidth_clamped_to_min() {
        let mut opt = RtmpChunkOptimizer::new();
        let size = opt.optimize_chunk_size(0);
        assert_eq!(size, MIN_CHUNK_SIZE);
    }

    #[test]
    fn test_optimize_chunk_size_stored() {
        let mut opt = RtmpChunkOptimizer::new();
        let size = opt.optimize_chunk_size(10_000_000);
        assert_eq!(opt.current_size(), size);
    }

    #[test]
    fn test_with_bounds_clamps() {
        let opt = RtmpChunkOptimizer::with_bounds(0, 1_000_000);
        assert_eq!(opt.min_chunk_size, MIN_CHUNK_SIZE);
        assert_eq!(opt.max_chunk_size, MAX_CHUNK_SIZE);
    }

    #[test]
    fn test_with_bounds_custom() {
        let opt = RtmpChunkOptimizer::with_bounds(256, 8192);
        assert_eq!(opt.min_chunk_size, 256);
        assert_eq!(opt.max_chunk_size, 8192);
    }

    // ── RtmpSessionStats ─────────────────────────────────────────────────────

    #[test]
    fn test_rtmp_session_stats_record_chunk() {
        let mut s = RtmpSessionStats::new();
        s.record_chunk(1024);
        assert_eq!(s.total_bytes, 1024);
        assert_eq!(s.chunks_sent, 1);
    }

    #[test]
    fn test_rtmp_session_stats_avg_chunk_size() {
        let mut s = RtmpSessionStats::new();
        s.record_chunk(1000);
        s.record_chunk(3000);
        // total = 4000, chunks = 2 → avg = 2000.0
        assert!((s.avg_chunk_size - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn test_rtmp_session_stats_drop_frame() {
        let mut s = RtmpSessionStats::new();
        s.drop_frame();
        s.drop_frame();
        assert_eq!(s.dropped_frames, 2);
    }

    #[test]
    fn test_rtmp_session_stats_reset() {
        let mut s = RtmpSessionStats::new();
        s.record_chunk(512);
        s.drop_frame();
        s.reset();
        assert_eq!(s.total_bytes, 0);
        assert_eq!(s.chunks_sent, 0);
        assert!((s.avg_chunk_size).abs() < 1e-12);
        assert_eq!(s.dropped_frames, 0);
    }

    #[test]
    fn test_rtmp_session_stats_throughput_bps() {
        let mut s = RtmpSessionStats::new();
        s.record_chunk(1_000_000); // 1 MB
        let bps = s.throughput_bps(1.0);
        assert!(
            (bps - 8_000_000.0).abs() < 1.0,
            "Expected 8 Mbps for 1 MB in 1 s"
        );
    }

    #[test]
    fn test_rtmp_session_stats_throughput_zero_elapsed() {
        let mut s = RtmpSessionStats::new();
        s.record_chunk(100);
        assert_eq!(s.throughput_bps(0.0), 0.0);
        assert_eq!(s.throughput_bps(-1.0), 0.0);
    }

    // ── adaptive_update ──────────────────────────────────────────────────────

    #[test]
    fn test_adaptive_update_high_loss_reduces_size() {
        let mut opt = RtmpChunkOptimizer::new();
        opt.current_chunk_size = 4096;
        let new_size = opt.adaptive_update(10_000_000, 6.0);
        // 4096 * 0.75 = 3072
        assert_eq!(new_size, 3072);
        assert_eq!(opt.current_size(), 3072);
    }

    #[test]
    fn test_adaptive_update_low_loss_optimizes() {
        let mut opt = RtmpChunkOptimizer::new();
        let new_size = opt.adaptive_update(10_000_000, 0.5);
        // Should use optimize_chunk_size → 384
        assert_eq!(new_size, 384);
    }

    #[test]
    fn test_adaptive_update_medium_loss_holds() {
        let mut opt = RtmpChunkOptimizer::new();
        opt.current_chunk_size = 2048;
        let new_size = opt.adaptive_update(10_000_000, 3.0);
        assert_eq!(new_size, 2048, "Medium loss should hold current size");
    }

    // ── suggested_chunk_count ────────────────────────────────────────────────

    #[test]
    fn test_suggested_chunk_count_exact() {
        let mut opt = RtmpChunkOptimizer::new();
        opt.current_chunk_size = 1024;
        assert_eq!(opt.suggested_chunk_count(4096), 4);
    }

    #[test]
    fn test_suggested_chunk_count_ceiling() {
        let mut opt = RtmpChunkOptimizer::new();
        opt.current_chunk_size = 1000;
        // 2500 / 1000 = 2.5 → ceil = 3
        assert_eq!(opt.suggested_chunk_count(2500), 3);
    }

    #[test]
    fn test_suggested_chunk_count_zero_bytes() {
        let opt = RtmpChunkOptimizer::new();
        assert_eq!(opt.suggested_chunk_count(0), 0);
    }

    // ── record_sent delegates to stats ───────────────────────────────────────

    #[test]
    fn test_record_sent_delegates() {
        let mut opt = RtmpChunkOptimizer::new();
        opt.record_sent(512);
        opt.record_sent(512);
        assert_eq!(opt.stats().total_bytes, 1024);
        assert_eq!(opt.stats().chunks_sent, 2);
    }
}
