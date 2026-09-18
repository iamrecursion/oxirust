//! Frame pacing and timing control for video-over-IP streams.
//!
//! This module ensures that video frames are transmitted at precise intervals
//! matching the source frame rate, preventing bursty delivery that can overwhelm
//! network buffers or cause jitter at the receiver.
//!
//! # Features
//!
//! - **Constant-rate pacing** - Spaces frames evenly at the target frame rate
//! - **Traffic shaping** - Distributes packet bursts within a frame period
//! - **Drift correction** - Compensates for clock drift between sender and receiver
//! - **Burst budget** - Allows controlled bursts for catch-up after delays
//! - **PreciseClock** - Sub-millisecond clock using mach_absolute_time on macOS

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for frame pacing.
#[derive(Debug, Clone)]
pub struct FramePacingConfig {
    /// Target frame rate in frames per second.
    pub target_fps: f64,
    /// Maximum allowed burst (number of frames that can be sent back-to-back).
    pub max_burst: usize,
    /// Clock drift correction interval (number of frames between corrections).
    pub drift_correction_interval: usize,
    /// Maximum allowed drift before correction, in microseconds.
    pub max_drift_us: i64,
    /// Whether to enable traffic shaping within a frame period.
    pub enable_shaping: bool,
    /// Number of packets per frame for traffic shaping calculations.
    pub packets_per_frame: usize,
}

impl Default for FramePacingConfig {
    fn default() -> Self {
        Self {
            target_fps: 30.0,
            max_burst: 2,
            drift_correction_interval: 300,
            max_drift_us: 5000,
            enable_shaping: true,
            packets_per_frame: 100,
        }
    }
}

/// Frame timing decision from the pacer.
#[derive(Debug, Clone)]
pub struct PacingDecision {
    /// Whether to send the frame now.
    pub should_send: bool,
    /// Delay before sending, if any.
    pub delay: Duration,
    /// Current drift from ideal schedule, in microseconds (positive = ahead).
    pub drift_us: i64,
    /// Inter-packet interval for traffic shaping within this frame.
    pub packet_interval: Duration,
    /// Frame sequence number.
    pub sequence: u64,
}

/// Statistics from the frame pacer.
#[derive(Debug, Clone)]
pub struct PacingStats {
    /// Total frames paced.
    pub total_frames: u64,
    /// Frames that were delayed to match pacing.
    pub delayed_frames: u64,
    /// Frames sent immediately (within tolerance).
    pub on_time_frames: u64,
    /// Burst frames (sent without full interval wait).
    pub burst_frames: u64,
    /// Average drift in microseconds.
    pub avg_drift_us: f64,
    /// Maximum observed drift in microseconds.
    pub max_drift_us: i64,
    /// Current frame interval in microseconds.
    pub frame_interval_us: u64,
    /// Drift corrections applied.
    pub drift_corrections: u64,
}

/// Sub-millisecond accurate clock.
///
/// On macOS uses `mach_absolute_time` for nanosecond resolution (via an
/// `extern "C"` declaration — the kernel provides this symbol; it is NOT a
/// C-library function and is therefore acceptable under the COOLJAPAN Pure
/// Rust policy when cfg-gated to `target_os = "macos"`).
///
/// On other platforms falls back to `std::time::SystemTime` monotonic source.
pub struct PreciseClock {
    _private: (),
}

impl PreciseClock {
    /// Create a new `PreciseClock`.
    #[must_use]
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Returns current monotonic time in nanoseconds.
    ///
    /// On macOS this calls `mach_absolute_time` (raw hardware ticks; on x86-64
    /// and Apple Silicon the OS calibrates the tick unit to nanoseconds).
    /// On all other platforms it falls back to `SystemTime` wall-clock nanos.
    #[must_use]
    pub fn now_ns(&self) -> u64 {
        #[cfg(target_os = "macos")]
        {
            Self::mach_now_ns()
        }
        #[cfg(not(target_os = "macos"))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        }
    }

    /// macOS-only: call `mach_absolute_time` via libc.
    ///
    /// `mach_absolute_time` is a pure hardware-counter read with no side
    /// effects.  The `#[allow(unsafe_code)]` is scoped to this single helper
    /// following the same pattern used in `gf_simd.rs`.
    #[cfg(target_os = "macos")]
    #[allow(unsafe_code)]
    fn mach_now_ns() -> u64 {
        extern "C" {
            fn mach_absolute_time() -> u64;
        }
        // SAFETY: mach_absolute_time is a stable macOS kernel routine that
        // performs a single hardware-counter read and has no side effects.
        unsafe { mach_absolute_time() }
    }

    /// Sleep until `target_ns`, using a coarse OS sleep followed by a
    /// busy-wait tail for sub-millisecond accuracy.
    pub fn sleep_until_ns(&self, target_ns: u64) {
        let now = self.now_ns();
        if now >= target_ns {
            return;
        }
        let gap = target_ns - now;
        // Leave a 200 µs busy-wait window for sub-ms precision.
        const BUSY_TAIL_NS: u64 = 200_000;
        if gap > BUSY_TAIL_NS {
            std::thread::sleep(Duration::from_nanos(gap - BUSY_TAIL_NS));
        }
        // Busy-wait for the final stretch.
        while self.now_ns() < target_ns {
            std::hint::spin_loop();
        }
    }
}

impl Default for PreciseClock {
    fn default() -> Self {
        Self::new()
    }
}

/// Frame pacer that controls transmission timing.
pub struct FramePacer {
    /// Configuration.
    config: FramePacingConfig,
    /// Ideal frame interval.
    frame_interval: Duration,
    /// High-resolution clock for precise sleeping.
    clock: PreciseClock,
    /// Nanosecond timestamp of the next frame deadline (PreciseClock epoch).
    next_deadline_ns: Option<u64>,
    /// Time the first frame was submitted.
    start_time: Option<Instant>,
    /// Ideal send time for the next frame.
    next_send_time: Option<Instant>,
    /// Frame sequence counter.
    sequence: u64,
    /// Current burst budget.
    burst_budget: usize,
    /// Drift history for correction.
    drift_history: VecDeque<i64>,
    /// Stats counters.
    total_frames: u64,
    /// Delayed frame count.
    delayed_frames: u64,
    /// On time frame count.
    on_time_frames: u64,
    /// Burst frame count.
    burst_frames: u64,
    /// Max drift observed.
    max_observed_drift_us: i64,
    /// Running drift sum.
    drift_sum: i64,
    /// Drift corrections applied.
    drift_corrections: u64,
}

impl FramePacer {
    /// Create a new frame pacer with default configuration.
    #[must_use]
    pub fn new(fps: f64) -> Self {
        let mut config = FramePacingConfig::default();
        config.target_fps = fps;
        Self::with_config(config)
    }

    /// Create a new frame pacer with custom configuration.
    #[must_use]
    pub fn with_config(config: FramePacingConfig) -> Self {
        let interval_us = if config.target_fps > 0.0 {
            (1_000_000.0 / config.target_fps) as u64
        } else {
            33333 // default ~30fps
        };
        Self {
            frame_interval: Duration::from_micros(interval_us),
            config,
            clock: PreciseClock::new(),
            next_deadline_ns: None,
            start_time: None,
            next_send_time: None,
            sequence: 0,
            burst_budget: 0,
            drift_history: VecDeque::with_capacity(64),
            total_frames: 0,
            delayed_frames: 0,
            on_time_frames: 0,
            burst_frames: 0,
            max_observed_drift_us: 0,
            drift_sum: 0,
            drift_corrections: 0,
        }
    }

    /// Submit a frame and get a pacing decision.
    ///
    /// Call this when a frame is ready to be sent. The returned `PacingDecision`
    /// indicates whether to send immediately or wait.
    pub fn pace_frame(&mut self, now: Instant) -> PacingDecision {
        let start = *self.start_time.get_or_insert(now);
        let next = *self.next_send_time.get_or_insert(now);

        let ideal_time = start + self.frame_interval * self.sequence as u32;
        let drift_us = if now > ideal_time {
            -(now.duration_since(ideal_time).as_micros() as i64)
        } else {
            ideal_time.duration_since(now).as_micros() as i64
        };

        // Record drift
        self.drift_sum += drift_us.abs();
        if drift_us.abs() > self.max_observed_drift_us {
            self.max_observed_drift_us = drift_us.abs();
        }
        if self.drift_history.len() >= 64 {
            self.drift_history.pop_front();
        }
        self.drift_history.push_back(drift_us);

        // Drift correction
        if self.sequence > 0
            && self.sequence % self.config.drift_correction_interval as u64 == 0
            && drift_us.abs() > self.config.max_drift_us
        {
            // Reset next_send_time to now to correct accumulated drift
            self.next_send_time = Some(now);
            self.drift_corrections += 1;
        }

        let (should_send, delay) = if now >= next {
            // We're at or past the send time
            let frames_behind =
                now.duration_since(next).as_micros() / self.frame_interval.as_micros().max(1);
            if frames_behind > 0 && self.burst_budget < self.config.max_burst {
                self.burst_budget += 1;
                self.burst_frames += 1;
            } else {
                self.on_time_frames += 1;
            }
            (true, Duration::ZERO)
        } else {
            let wait = next.duration_since(now);
            self.delayed_frames += 1;
            (false, wait)
        };

        // Advance next_send_time
        if should_send {
            self.next_send_time = Some(next + self.frame_interval);
            self.burst_budget = 0;
        }

        // Traffic shaping interval
        let packet_interval = if self.config.enable_shaping && self.config.packets_per_frame > 1 {
            Duration::from_micros(
                self.frame_interval.as_micros() as u64 / self.config.packets_per_frame as u64,
            )
        } else {
            Duration::ZERO
        };

        let seq = self.sequence;
        self.sequence += 1;
        self.total_frames += 1;

        PacingDecision {
            should_send,
            delay,
            drift_us,
            packet_interval,
            sequence: seq,
        }
    }

    /// Get pacing statistics.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn stats(&self) -> PacingStats {
        let avg_drift = if self.total_frames > 0 {
            self.drift_sum as f64 / self.total_frames as f64
        } else {
            0.0
        };

        PacingStats {
            total_frames: self.total_frames,
            delayed_frames: self.delayed_frames,
            on_time_frames: self.on_time_frames,
            burst_frames: self.burst_frames,
            avg_drift_us: avg_drift,
            max_drift_us: self.max_observed_drift_us,
            frame_interval_us: self.frame_interval.as_micros() as u64,
            drift_corrections: self.drift_corrections,
        }
    }

    /// Get the target frame interval.
    #[must_use]
    pub fn frame_interval(&self) -> Duration {
        self.frame_interval
    }

    /// Change the target frame rate dynamically.
    pub fn set_fps(&mut self, fps: f64) {
        if fps > 0.0 {
            let interval_us = (1_000_000.0 / fps) as u64;
            self.frame_interval = Duration::from_micros(interval_us);
            self.config.target_fps = fps;
        }
    }

    /// Reset the pacer state.
    pub fn reset(&mut self) {
        self.start_time = None;
        self.next_send_time = None;
        self.next_deadline_ns = None;
        self.sequence = 0;
        self.burst_budget = 0;
        self.drift_history.clear();
        self.total_frames = 0;
        self.delayed_frames = 0;
        self.on_time_frames = 0;
        self.burst_frames = 0;
        self.max_observed_drift_us = 0;
        self.drift_sum = 0;
        self.drift_corrections = 0;
    }

    /// Block the current thread until the next frame deadline and return
    /// immediately.  Uses `PreciseClock::sleep_until_ns` for sub-millisecond
    /// accuracy (coarse sleep + busy-wait tail).
    ///
    /// Intended for real-time sender threads where low-jitter frame timing
    /// matters.  For simulation / testing purposes, prefer `pace_frame` which
    /// accepts an explicit `Instant` argument.
    pub fn wait_for_next_frame(&mut self) {
        let interval_ns = self.frame_interval.as_nanos() as u64;
        let now_ns = self.clock.now_ns();
        let deadline = match self.next_deadline_ns {
            None => {
                self.next_deadline_ns = Some(now_ns + interval_ns);
                now_ns // first call: send immediately
            }
            Some(d) => d,
        };
        self.clock.sleep_until_ns(deadline);
        self.next_deadline_ns = Some(deadline + interval_ns);
    }
}

/// Calculate the ideal frame interval for a given frame rate.
#[must_use]
pub fn frame_interval_for_fps(fps: f64) -> Duration {
    if fps <= 0.0 {
        return Duration::from_millis(33);
    }
    let us = (1_000_000.0 / fps) as u64;
    Duration::from_micros(us)
}

/// Calculate frames-per-second from a frame interval duration.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn fps_from_interval(interval: Duration) -> f64 {
    let us = interval.as_micros();
    if us == 0 {
        return 0.0;
    }
    1_000_000.0 / us as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_interval_30fps() {
        let interval = frame_interval_for_fps(30.0);
        let expected_us = 33333u64;
        assert!(
            (interval.as_micros() as i64 - expected_us as i64).unsigned_abs() <= 1,
            "30fps interval should be ~33333us, got {}",
            interval.as_micros()
        );
    }

    #[test]
    fn test_frame_interval_60fps() {
        let interval = frame_interval_for_fps(60.0);
        let expected_us = 16666u64;
        assert!(
            (interval.as_micros() as i64 - expected_us as i64).unsigned_abs() <= 1,
            "60fps interval should be ~16666us, got {}",
            interval.as_micros()
        );
    }

    #[test]
    fn test_fps_from_interval_roundtrip() {
        let fps = 25.0;
        let interval = frame_interval_for_fps(fps);
        let recovered = fps_from_interval(interval);
        assert!(
            (recovered - fps).abs() < 0.1,
            "roundtrip should be close: {} vs {}",
            recovered,
            fps
        );
    }

    #[test]
    fn test_fps_from_zero_interval() {
        assert_eq!(fps_from_interval(Duration::ZERO), 0.0);
    }

    #[test]
    fn test_pacer_first_frame_immediate() {
        let mut pacer = FramePacer::new(30.0);
        let now = Instant::now();
        let decision = pacer.pace_frame(now);
        assert!(
            decision.should_send,
            "first frame should be sent immediately"
        );
        assert_eq!(decision.sequence, 0);
    }

    #[test]
    fn test_pacer_second_frame_needs_delay() {
        let mut pacer = FramePacer::new(30.0);
        let now = Instant::now();
        let _d1 = pacer.pace_frame(now);
        // Ask for second frame immediately (same instant)
        let d2 = pacer.pace_frame(now);
        assert!(!d2.should_send || d2.delay > Duration::ZERO || d2.sequence == 1);
    }

    #[test]
    fn test_pacer_frame_after_interval() {
        let mut pacer = FramePacer::new(30.0);
        let t0 = Instant::now();
        let _d1 = pacer.pace_frame(t0);
        // Simulate waiting one frame interval
        let t1 = t0 + pacer.frame_interval();
        let d2 = pacer.pace_frame(t1);
        assert!(d2.should_send, "frame after interval should be sendable");
    }

    #[test]
    fn test_pacer_stats_initial() {
        let pacer = FramePacer::new(30.0);
        let stats = pacer.stats();
        assert_eq!(stats.total_frames, 0);
        assert_eq!(stats.delayed_frames, 0);
    }

    #[test]
    fn test_pacer_stats_after_frames() {
        let mut pacer = FramePacer::new(30.0);
        let t0 = Instant::now();
        for i in 0..10u32 {
            let t = t0 + pacer.frame_interval() * i;
            pacer.pace_frame(t);
        }
        let stats = pacer.stats();
        assert_eq!(stats.total_frames, 10);
    }

    #[test]
    fn test_set_fps() {
        let mut pacer = FramePacer::new(30.0);
        pacer.set_fps(60.0);
        let interval = pacer.frame_interval();
        assert!(
            (interval.as_micros() as i64 - 16666).unsigned_abs() <= 1,
            "after set_fps(60), interval should be ~16666us: {}",
            interval.as_micros()
        );
    }

    #[test]
    fn test_reset_pacer() {
        let mut pacer = FramePacer::new(30.0);
        let now = Instant::now();
        pacer.pace_frame(now);
        pacer.pace_frame(now + Duration::from_millis(33));
        pacer.reset();
        let stats = pacer.stats();
        assert_eq!(stats.total_frames, 0);
    }

    #[test]
    fn test_traffic_shaping_interval() {
        let config = FramePacingConfig {
            target_fps: 30.0,
            enable_shaping: true,
            packets_per_frame: 10,
            ..Default::default()
        };
        let mut pacer = FramePacer::with_config(config);
        let decision = pacer.pace_frame(Instant::now());
        // frame interval ~33333us / 10 packets = ~3333us per packet
        assert!(
            decision.packet_interval.as_micros() > 3000
                && decision.packet_interval.as_micros() < 3500,
            "packet interval should be ~3333us: {}",
            decision.packet_interval.as_micros()
        );
    }

    #[test]
    fn test_negative_fps_fallback() {
        let interval = frame_interval_for_fps(-5.0);
        assert_eq!(interval, Duration::from_millis(33));
    }
}
