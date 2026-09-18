//! SMPTE ST 2110-21 traffic shaping and delivery timing.
//!
//! This module implements SMPTE ST 2110-21 which defines timing and synchronization
//! for SMPTE ST 2110 streams, including narrow/wide timing modes, gapped/linear
//! transmission, PTP synchronization, and jitter handling.

use crate::error::{NetError, NetResult};
use std::time::{Duration, Instant};

/// Timing compliance mode as defined in ST 2110-21.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingMode {
    /// Narrow timing mode - strictest timing requirements.
    /// Used for low-latency applications.
    Narrow,
    /// Wide timing mode - relaxed timing requirements.
    /// Used for standard broadcast applications.
    Wide,
    /// Wide-linear timing mode - wide timing with linear transmission.
    WideLinear,
}

/// Transmission pattern for RTP packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransmissionMode {
    /// Gapped transmission - packets sent in bursts with gaps.
    Gapped,
    /// Linear transmission - packets sent at constant rate.
    Linear,
}

/// Video scan type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanType {
    /// Progressive scan.
    Progressive,
    /// Interlaced scan - Field 1.
    InterlacedField1,
    /// Interlaced scan - Field 2.
    InterlacedField2,
}

/// Frame rate specification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameRate {
    /// Numerator of frame rate fraction.
    pub numerator: u32,
    /// Denominator of frame rate fraction.
    pub denominator: u32,
}

impl FrameRate {
    /// Creates a new frame rate.
    #[must_use]
    pub const fn new(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Returns the frame rate as a floating point value.
    #[must_use]
    pub fn as_f64(&self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator)
    }

    /// Returns the frame period in nanoseconds.
    #[must_use]
    pub fn frame_period_ns(&self) -> u64 {
        (1_000_000_000u64 * u64::from(self.denominator)) / u64::from(self.numerator)
    }

    /// Common frame rates.
    pub const FPS_23_976: Self = Self::new(24000, 1001);
    /// 24 frames per second.
    pub const FPS_24: Self = Self::new(24, 1);
    /// 25 frames per second (PAL).
    pub const FPS_25: Self = Self::new(25, 1);
    /// 29.97 frames per second (NTSC).
    pub const FPS_29_97: Self = Self::new(30000, 1001);
    /// 30 frames per second.
    pub const FPS_30: Self = Self::new(30, 1);
    /// 50 frames per second (PAL interlaced).
    pub const FPS_50: Self = Self::new(50, 1);
    /// 59.94 frames per second (NTSC interlaced).
    pub const FPS_59_94: Self = Self::new(60000, 1001);
    /// 60 frames per second.
    pub const FPS_60: Self = Self::new(60, 1);
    /// 100 frames per second.
    pub const FPS_100: Self = Self::new(100, 1);
    /// 119.88 frames per second.
    pub const FPS_119_88: Self = Self::new(120000, 1001);
    /// 120 frames per second.
    pub const FPS_120: Self = Self::new(120, 1);
}

/// Timing parameters for narrow mode (ST 2110-21 Section 7.1).
#[derive(Debug, Clone)]
pub struct NarrowTimingParams {
    /// T_READINESS - time before first packet of frame can be transmitted.
    pub t_readiness_ns: u64,
    /// T_DRAIN - time by which all packets must be transmitted.
    pub t_drain_ns: u64,
}

/// Timing parameters for wide mode (ST 2110-21 Section 7.2).
#[derive(Debug, Clone)]
pub struct WideTimingParams {
    /// T_READINESS - time before first packet can be transmitted.
    pub t_readiness_ns: u64,
    /// T_DRAIN - time by which all packets must be transmitted.
    pub t_drain_ns: u64,
}

/// ST 2110-21 timing calculator.
#[derive(Debug, Clone)]
pub struct TimingCalculator {
    /// Timing mode.
    mode: TimingMode,
    /// Transmission mode.
    transmission: TransmissionMode,
    /// Frame rate.
    frame_rate: FrameRate,
    /// Active line count (video).
    active_lines: u32,
    /// Packets per frame.
    packets_per_frame: u32,
}

impl TimingCalculator {
    /// Creates a new timing calculator.
    #[must_use]
    pub const fn new(
        mode: TimingMode,
        transmission: TransmissionMode,
        frame_rate: FrameRate,
        active_lines: u32,
        packets_per_frame: u32,
    ) -> Self {
        Self {
            mode,
            transmission,
            frame_rate,
            active_lines,
            packets_per_frame,
        }
    }

    /// Calculates T_READINESS for narrow mode.
    ///
    /// T_READINESS = (1/frame_rate) - T_DRAIN
    #[must_use]
    pub fn narrow_t_readiness(&self) -> u64 {
        let frame_period = self.frame_rate.frame_period_ns();
        let t_drain = self.narrow_t_drain();
        frame_period.saturating_sub(t_drain)
    }

    /// Calculates T_DRAIN for narrow mode.
    ///
    /// For gapped: T_DRAIN = active_lines * line_time
    /// For linear: T_DRAIN = frame_time - (43 * line_time)
    #[must_use]
    pub fn narrow_t_drain(&self) -> u64 {
        let frame_period = self.frame_rate.frame_period_ns();
        let total_lines: u64 = if self.frame_rate.numerator == 25 || self.frame_rate.numerator == 50
        {
            625 // PAL
        } else {
            525 // NTSC
        };
        let line_time = frame_period / total_lines;

        match self.transmission {
            TransmissionMode::Gapped => u64::from(self.active_lines) * line_time,
            TransmissionMode::Linear => frame_period.saturating_sub(43 * line_time),
        }
    }

    /// Calculates T_READINESS for wide mode.
    ///
    /// T_READINESS = (1/frame_rate) - T_DRAIN
    #[must_use]
    pub fn wide_t_readiness(&self) -> u64 {
        let frame_period = self.frame_rate.frame_period_ns();
        let t_drain = self.wide_t_drain();
        frame_period.saturating_sub(t_drain)
    }

    /// Calculates T_DRAIN for wide mode.
    ///
    /// Wide mode allows entire frame period for transmission.
    #[must_use]
    pub fn wide_t_drain(&self) -> u64 {
        self.frame_rate.frame_period_ns()
    }

    /// Gets the timing parameters based on current mode.
    #[must_use]
    pub fn get_timing_params(&self) -> (u64, u64) {
        match self.mode {
            TimingMode::Narrow => (self.narrow_t_readiness(), self.narrow_t_drain()),
            TimingMode::Wide | TimingMode::WideLinear => {
                (self.wide_t_readiness(), self.wide_t_drain())
            }
        }
    }

    /// Calculates the packet transmission time for a given packet index.
    ///
    /// For gapped mode, packets are sent in line-based bursts.
    /// For linear mode, packets are sent at constant intervals.
    #[must_use]
    pub fn packet_transmission_time(&self, packet_index: u32, frame_start_ns: u64) -> u64 {
        let (t_readiness, t_drain) = self.get_timing_params();

        match self.transmission {
            TransmissionMode::Gapped => {
                // Packets sent in bursts per line
                let packets_per_line = self.packets_per_frame / self.active_lines;
                let line_index = packet_index / packets_per_line;
                let frame_period = self.frame_rate.frame_period_ns();
                let line_time = frame_period / u64::from(self.active_lines);

                frame_start_ns + t_readiness + (u64::from(line_index) * line_time)
            }
            TransmissionMode::Linear => {
                // Packets sent at constant rate
                let packet_interval = t_drain / u64::from(self.packets_per_frame);
                frame_start_ns + t_readiness + (u64::from(packet_index) * packet_interval)
            }
        }
    }
}

/// RTP timestamp generator for SMPTE ST 2110.
#[derive(Debug, Clone)]
pub struct RtpTimestampGenerator {
    /// Clock rate (Hz) - typically 90000 for video.
    clock_rate: u32,
    /// Current timestamp.
    current_timestamp: u32,
    /// Frame rate.
    frame_rate: FrameRate,
}

impl RtpTimestampGenerator {
    /// Creates a new RTP timestamp generator.
    #[must_use]
    pub fn new(clock_rate: u32, frame_rate: FrameRate, initial_timestamp: u32) -> Self {
        Self {
            clock_rate,
            current_timestamp: initial_timestamp,
            frame_rate,
        }
    }

    /// Gets the current timestamp.
    #[must_use]
    pub const fn current(&self) -> u32 {
        self.current_timestamp
    }

    /// Advances to the next frame timestamp.
    pub fn advance_frame(&mut self) {
        let increment = (u64::from(self.clock_rate) * u64::from(self.frame_rate.denominator))
            / u64::from(self.frame_rate.numerator);
        self.current_timestamp = self.current_timestamp.wrapping_add(increment as u32);
    }

    /// Converts PTP timestamp (nanoseconds) to RTP timestamp.
    #[must_use]
    pub fn ptp_to_rtp(&self, ptp_ns: u64) -> u32 {
        ((ptp_ns * u64::from(self.clock_rate)) / 1_000_000_000) as u32
    }

    /// Converts RTP timestamp to PTP timestamp (nanoseconds).
    #[must_use]
    pub fn rtp_to_ptp(&self, rtp_ts: u32) -> u64 {
        (u64::from(rtp_ts) * 1_000_000_000) / u64::from(self.clock_rate)
    }
}

/// Playout buffer for handling jitter and timing variations.
#[derive(Debug)]
pub struct PlayoutBuffer<T> {
    /// Buffered items with their target playout time.
    buffer: Vec<(u64, T)>,
    /// Target buffer depth in nanoseconds.
    target_depth_ns: u64,
    /// Maximum buffer size.
    max_size: usize,
}

impl<T> PlayoutBuffer<T> {
    /// Creates a new playout buffer.
    #[must_use]
    pub fn new(target_depth_ns: u64, max_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(max_size),
            target_depth_ns,
            max_size,
        }
    }

    /// Adds an item to the buffer with its playout time.
    pub fn push(&mut self, playout_time_ns: u64, item: T) -> NetResult<()> {
        if self.buffer.len() >= self.max_size {
            return Err(NetError::buffer("Playout buffer full"));
        }

        // Insert in sorted order by playout time
        let pos = self
            .buffer
            .binary_search_by_key(&playout_time_ns, |(t, _)| *t)
            .unwrap_or_else(|e| e);
        self.buffer.insert(pos, (playout_time_ns, item));

        Ok(())
    }

    /// Retrieves items ready for playout at the given time.
    pub fn pop_ready(&mut self, current_time_ns: u64) -> Vec<T> {
        let mut ready = Vec::new();

        while let Some((playout_time, _)) = self.buffer.first() {
            if *playout_time <= current_time_ns {
                let (_, item) = self.buffer.remove(0);
                ready.push(item);
            } else {
                break;
            }
        }

        ready
    }

    /// Gets the current buffer depth in nanoseconds.
    #[must_use]
    pub fn depth_ns(&self, current_time_ns: u64) -> u64 {
        if let Some((playout_time, _)) = self.buffer.last() {
            playout_time.saturating_sub(current_time_ns)
        } else {
            0
        }
    }

    /// Checks if buffer has reached target depth.
    #[must_use]
    pub fn is_ready(&self, current_time_ns: u64) -> bool {
        self.depth_ns(current_time_ns) >= self.target_depth_ns
    }

    /// Gets the number of items in buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Checks if buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Rate controller for linear transmission mode.
#[derive(Debug, Clone)]
pub struct RateController {
    /// Target bitrate (bits per second).
    target_bitrate: u64,
    /// Current time in nanoseconds.
    current_time_ns: u64,
    /// Bytes transmitted.
    bytes_transmitted: u64,
    /// Start time.
    start_time: Instant,
}

impl RateController {
    /// Creates a new rate controller.
    #[must_use]
    pub fn new(target_bitrate: u64) -> Self {
        Self {
            target_bitrate,
            current_time_ns: 0,
            bytes_transmitted: 0,
            start_time: Instant::now(),
        }
    }

    /// Calculates the delay needed before transmitting the next packet.
    #[must_use]
    pub fn calculate_delay(&self, packet_size: usize) -> Duration {
        let elapsed = self.start_time.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;

        let bits_transmitted = self.bytes_transmitted * 8;
        let expected_bits = (elapsed_ns * self.target_bitrate) / 1_000_000_000;

        if bits_transmitted < expected_bits {
            // We're behind schedule - send immediately
            Duration::ZERO
        } else {
            // Calculate delay to maintain target rate
            let next_bits = bits_transmitted + ((packet_size * 8) as u64);
            let next_time_ns = (next_bits * 1_000_000_000) / self.target_bitrate;
            let delay_ns = next_time_ns.saturating_sub(elapsed_ns);
            Duration::from_nanos(delay_ns)
        }
    }

    /// Records a transmitted packet.
    pub fn record_transmission(&mut self, packet_size: usize) {
        self.bytes_transmitted += packet_size as u64;
    }

    /// Resets the rate controller.
    pub fn reset(&mut self) {
        self.bytes_transmitted = 0;
        self.start_time = Instant::now();
    }
}

/// Synchronization state tracker.
#[derive(Debug, Clone)]
pub struct SyncState {
    /// Whether PTP is synchronized.
    pub ptp_synced: bool,
    /// Current PTP time offset in nanoseconds.
    pub ptp_offset_ns: i64,
    /// PTP clock accuracy in nanoseconds.
    pub ptp_accuracy_ns: u64,
    /// Last sync update time.
    pub last_sync_update: Instant,
}

impl SyncState {
    /// Creates a new synchronization state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ptp_synced: false,
            ptp_offset_ns: 0,
            ptp_accuracy_ns: 0,
            last_sync_update: Instant::now(),
        }
    }

    /// Updates the PTP synchronization state.
    pub fn update_ptp(&mut self, offset_ns: i64, accuracy_ns: u64) {
        self.ptp_offset_ns = offset_ns;
        self.ptp_accuracy_ns = accuracy_ns;
        self.ptp_synced = accuracy_ns < 1_000_000; // Synced if accuracy < 1ms
        self.last_sync_update = Instant::now();
    }

    /// Checks if synchronization is valid.
    #[must_use]
    pub fn is_synced(&self) -> bool {
        self.ptp_synced && self.last_sync_update.elapsed() < Duration::from_secs(10)
    }

    /// Gets the current PTP time estimate.
    #[must_use]
    pub fn current_ptp_time_ns(&self, system_time_ns: u64) -> u64 {
        if self.ptp_offset_ns >= 0 {
            system_time_ns + (self.ptp_offset_ns as u64)
        } else {
            system_time_ns.saturating_sub(self.ptp_offset_ns.unsigned_abs())
        }
    }
}

impl Default for SyncState {
    fn default() -> Self {
        Self::new()
    }
}

/// Frame synchronization point calculator.
#[derive(Debug, Clone)]
pub struct FrameSyncPoint {
    /// Frame rate.
    frame_rate: FrameRate,
    /// PTP epoch (reference point).
    ptp_epoch_ns: u64,
}

impl FrameSyncPoint {
    /// Creates a new frame sync point calculator.
    #[must_use]
    pub const fn new(frame_rate: FrameRate, ptp_epoch_ns: u64) -> Self {
        Self {
            frame_rate,
            ptp_epoch_ns,
        }
    }

    /// Calculates the frame start time for a given PTP timestamp.
    #[must_use]
    pub fn frame_start_time(&self, ptp_time_ns: u64) -> u64 {
        let frame_period_ns = self.frame_rate.frame_period_ns();
        let elapsed = ptp_time_ns.saturating_sub(self.ptp_epoch_ns);
        let frame_number = elapsed / frame_period_ns;
        self.ptp_epoch_ns + (frame_number * frame_period_ns)
    }

    /// Calculates the next frame start time after the given PTP timestamp.
    #[must_use]
    pub fn next_frame_start(&self, ptp_time_ns: u64) -> u64 {
        let current_frame_start = self.frame_start_time(ptp_time_ns);
        current_frame_start + self.frame_rate.frame_period_ns()
    }

    /// Calculates the frame number for a given PTP timestamp.
    #[must_use]
    pub fn frame_number(&self, ptp_time_ns: u64) -> u64 {
        let frame_period_ns = self.frame_rate.frame_period_ns();
        let elapsed = ptp_time_ns.saturating_sub(self.ptp_epoch_ns);
        elapsed / frame_period_ns
    }
}

/// Timing validator for ST 2110-21 compliance.
#[derive(Debug)]
pub struct TimingValidator {
    /// Timing mode.
    mode: TimingMode,
    /// Expected frame rate.
    frame_rate: FrameRate,
    /// Allowed timing deviation (nanoseconds).
    max_deviation_ns: u64,
}

impl TimingValidator {
    /// Creates a new timing validator.
    #[must_use]
    pub fn new(mode: TimingMode, frame_rate: FrameRate) -> Self {
        let max_deviation_ns = match mode {
            TimingMode::Narrow => 100_000,       // 100 microseconds
            TimingMode::Wide => 1_000_000,       // 1 millisecond
            TimingMode::WideLinear => 1_000_000, // 1 millisecond
        };

        Self {
            mode,
            frame_rate,
            max_deviation_ns,
        }
    }

    /// Validates packet timing against ST 2110-21 requirements.
    pub fn validate_packet_timing(
        &self,
        expected_time_ns: u64,
        actual_time_ns: u64,
    ) -> NetResult<()> {
        let deviation = if actual_time_ns > expected_time_ns {
            actual_time_ns - expected_time_ns
        } else {
            expected_time_ns - actual_time_ns
        };

        if deviation > self.max_deviation_ns {
            Err(NetError::protocol(format!(
                "Timing violation: deviation {} ns exceeds limit {} ns",
                deviation, self.max_deviation_ns
            )))
        } else {
            Ok(())
        }
    }

    /// Validates frame timing.
    pub fn validate_frame_timing(
        &self,
        expected_frame_start_ns: u64,
        actual_frame_start_ns: u64,
    ) -> NetResult<()> {
        let frame_period_ns = self.frame_rate.frame_period_ns();
        let max_frame_deviation = frame_period_ns / 1000; // 0.1% of frame period

        let deviation = if actual_frame_start_ns > expected_frame_start_ns {
            actual_frame_start_ns - expected_frame_start_ns
        } else {
            expected_frame_start_ns - actual_frame_start_ns
        };

        if deviation > max_frame_deviation {
            Err(NetError::protocol(format!(
                "Frame timing violation: deviation {} ns exceeds limit {} ns",
                deviation, max_frame_deviation
            )))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_rate() {
        let fps_25 = FrameRate::FPS_25;
        assert_eq!(fps_25.as_f64(), 25.0);
        assert_eq!(fps_25.frame_period_ns(), 40_000_000);

        let fps_29_97 = FrameRate::FPS_29_97;
        assert!((fps_29_97.as_f64() - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_timing_calculator_narrow() {
        let calc = TimingCalculator::new(
            TimingMode::Narrow,
            TransmissionMode::Gapped,
            FrameRate::FPS_25,
            576,
            576,
        );

        let t_readiness = calc.narrow_t_readiness();
        let t_drain = calc.narrow_t_drain();

        assert!(t_readiness > 0);
        assert!(t_drain > 0);
        assert!(t_readiness + t_drain <= FrameRate::FPS_25.frame_period_ns());
    }

    #[test]
    fn test_rtp_timestamp_generator() {
        let mut gen = RtpTimestampGenerator::new(90000, FrameRate::FPS_25, 0);

        let ts1 = gen.current();
        gen.advance_frame();
        let ts2 = gen.current();

        assert_eq!(ts2 - ts1, 3600); // 90000/25 = 3600
    }

    #[test]
    fn test_playout_buffer() {
        let mut buffer = PlayoutBuffer::new(1_000_000, 100);

        buffer.push(100, "packet1").expect("should succeed in test");
        buffer.push(200, "packet2").expect("should succeed in test");
        buffer.push(150, "packet3").expect("should succeed in test");

        let ready = buffer.pop_ready(125);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], "packet1");

        let ready = buffer.pop_ready(200);
        assert_eq!(ready.len(), 2);
    }

    #[test]
    fn test_sync_state() {
        let mut state = SyncState::new();
        assert!(!state.is_synced());

        state.update_ptp(1000, 100_000);
        assert!(state.is_synced());
    }

    #[test]
    fn test_frame_sync_point() {
        let sync = FrameSyncPoint::new(FrameRate::FPS_25, 0);

        let frame_start = sync.frame_start_time(45_000_000);
        assert_eq!(frame_start, 40_000_000); // Frame 1

        let frame_num = sync.frame_number(45_000_000);
        assert_eq!(frame_num, 1);
    }
}
