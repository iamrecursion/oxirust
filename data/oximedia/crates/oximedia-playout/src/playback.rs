//! Real-time playback engine with frame-accurate timing
//!
//! Provides real-time playback with frame-accurate output, clock synchronization,
//! genlock support, buffer management, and emergency fallback.

use crate::{AudioFormat, PlayoutConfig, PlayoutError, Result, VideoFormat};
use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration as StdDuration, Instant};
use tokio::sync::{mpsc, Mutex};

/// Playback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybackConfig {
    /// Video format
    pub video_format: VideoFormat,

    /// Audio format
    pub audio_format: AudioFormat,

    /// Buffer size in frames
    pub buffer_size: usize,

    /// Genlock enabled
    pub genlock_enabled: bool,

    /// Clock source
    pub clock_source: ClockSource,

    /// Pre-roll frames
    pub preroll_frames: u32,

    /// Maximum frame drop tolerance
    pub max_frame_drops: u32,

    /// Emergency fallback path
    pub fallback_path: PathBuf,

    /// Enable frame drop detection
    pub detect_drops: bool,

    /// Maximum latency in milliseconds
    pub max_latency_ms: u64,
}

impl PlaybackConfig {
    /// Create from playout config
    pub fn from_playout_config(config: &PlayoutConfig) -> Self {
        Self {
            video_format: config.video_format.clone(),
            audio_format: config.audio_format.clone(),
            buffer_size: config.buffer_size,
            genlock_enabled: config.genlock_enabled,
            clock_source: ClockSource::from_string(&config.clock_source),
            preroll_frames: 5,
            max_frame_drops: 3,
            fallback_path: config.fallback_content.clone(),
            detect_drops: config.detect_frame_drops,
            max_latency_ms: config.max_latency_ms,
        }
    }
}

/// Clock source for synchronization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClockSource {
    /// Internal system clock
    Internal,
    /// SDI input reference
    SDI,
    /// Precision Time Protocol
    PTP,
    /// Network Time Protocol
    NTP,
    /// Genlock reference
    Genlock,
}

impl ClockSource {
    fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "sdi" => Self::SDI,
            "ptp" => Self::PTP,
            "ntp" => Self::NTP,
            "genlock" => Self::Genlock,
            _ => Self::Internal,
        }
    }
}

/// Frame buffer containing video and audio data
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    /// Frame number
    pub frame_number: u64,

    /// Presentation timestamp
    pub pts: DateTime<Utc>,

    /// Video data (placeholder for actual frame data)
    pub video_data: Vec<u8>,

    /// Audio data (placeholder for actual audio samples)
    pub audio_data: Vec<u8>,

    /// Frame width
    pub width: u32,

    /// Frame height
    pub height: u32,

    /// Audio sample count
    pub audio_samples: usize,

    /// Metadata
    pub metadata: FrameMetadata,
}

/// Frame metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrameMetadata {
    /// Source file path
    pub source_path: Option<PathBuf>,

    /// Timecode
    pub timecode: Option<String>,

    /// Field order (for interlaced)
    pub field_order: FieldOrder,

    /// Color space
    pub color_space: ColorSpace,

    /// Flags
    pub flags: FrameFlags,
}

/// Field order for interlaced content
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FieldOrder {
    #[default]
    Progressive,
    TopFieldFirst,
    BottomFieldFirst,
}

/// Color space information
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ColorSpace {
    #[default]
    BT709,
    BT601,
    BT2020,
    SRGB,
}

/// Frame flags
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrameFlags {
    /// Keyframe flag
    pub keyframe: bool,

    /// Dropped frame flag
    pub dropped: bool,

    /// Late frame flag
    pub late: bool,

    /// Emergency fallback flag
    pub fallback: bool,
}

/// Playback state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    Stopped,
    Playing,
    Paused,
    Buffering,
    Fallback,
}

/// Playback statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlaybackStats {
    /// Total frames played
    pub frames_played: u64,

    /// Frames dropped
    pub frames_dropped: u64,

    /// Frames late
    pub frames_late: u64,

    /// Average latency in microseconds
    pub avg_latency_us: u64,

    /// Maximum latency in microseconds
    pub max_latency_us: u64,

    /// Buffer underruns
    pub buffer_underruns: u64,

    /// Clock drift in microseconds
    pub clock_drift_us: i64,

    /// Uptime in seconds
    pub uptime_seconds: u64,
}

impl PlaybackStats {
    /// Calculate frame drop rate
    pub fn drop_rate(&self) -> f64 {
        if self.frames_played == 0 {
            0.0
        } else {
            (self.frames_dropped as f64 / self.frames_played as f64) * 100.0
        }
    }

    /// Calculate late frame rate
    pub fn late_rate(&self) -> f64 {
        if self.frames_played == 0 {
            0.0
        } else {
            (self.frames_late as f64 / self.frames_played as f64) * 100.0
        }
    }
}

/// Clock synchronization state
#[derive(Debug, Clone)]
struct ClockState {
    /// Reference time
    reference_time: Instant,

    /// UTC reference
    utc_reference: DateTime<Utc>,

    /// Frame counter
    frame_counter: u64,

    /// Drift compensation
    drift_us: i64,

    /// Last sync time
    last_sync: Instant,
}

impl ClockState {
    fn new() -> Self {
        Self {
            reference_time: Instant::now(),
            utc_reference: Utc::now(),
            frame_counter: 0,
            drift_us: 0,
            last_sync: Instant::now(),
        }
    }

    /// Calculate current frame number
    fn current_frame(&self, frame_rate: f64) -> u64 {
        let elapsed = self.reference_time.elapsed();
        let seconds = elapsed.as_secs_f64();
        (seconds * frame_rate) as u64
    }

    /// Calculate next frame time
    fn next_frame_time(&self, frame_rate: f64) -> Instant {
        let frame_duration_us = (1_000_000.0 / frame_rate) as u64;
        let next_frame = self.frame_counter + 1;
        let total_us = next_frame * frame_duration_us;
        let drift_adjusted = (total_us as i64 + self.drift_us) as u64;

        self.reference_time + StdDuration::from_micros(drift_adjusted)
    }
}

/// Buffer manager for frame buffering
struct BufferManager {
    /// Frame buffer queue
    buffer: VecDeque<FrameBuffer>,

    /// Maximum buffer size
    max_size: usize,

    /// Minimum buffer level before underrun
    min_level: usize,

    /// Buffer state
    state: BufferState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum BufferState {
    Empty,
    Filling,
    Ready,
    Underrun,
}

impl BufferManager {
    fn new(max_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_size),
            max_size,
            min_level: max_size / 2,
            state: BufferState::Empty,
        }
    }

    /// Push a frame to the buffer
    fn push(&mut self, frame: FrameBuffer) -> Result<()> {
        if self.buffer.len() >= self.max_size {
            return Err(PlayoutError::Playback("Buffer full".to_string()));
        }

        self.buffer.push_back(frame);
        self.update_state();

        Ok(())
    }

    /// Pop a frame from the buffer
    fn pop(&mut self) -> Option<FrameBuffer> {
        let frame = self.buffer.pop_front();
        self.update_state();
        frame
    }

    /// Get current buffer level
    fn level(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is ready
    fn is_ready(&self) -> bool {
        self.state == BufferState::Ready
    }

    /// Update buffer state
    fn update_state(&mut self) {
        self.state = if self.buffer.is_empty() {
            BufferState::Empty
        } else if self.buffer.len() < self.min_level {
            BufferState::Filling
        } else {
            BufferState::Ready
        };
    }

    /// Clear buffer
    fn clear(&mut self) {
        self.buffer.clear();
        self.state = BufferState::Empty;
    }
}

/// Genlock status report
#[derive(Debug, Clone)]
pub struct GenlockStatus {
    /// Whether the output is locked to the reference signal
    pub is_locked: bool,
    /// Phase error between output and reference in microseconds
    pub phase_error_us: f64,
    /// Frequency error in parts-per-million
    pub frequency_error_ppm: f64,
}

/// Genlock synchronizer
struct GenlockSync {
    /// Genlock enabled
    enabled: bool,

    /// Reference signal detected
    reference_detected: bool,

    /// Lock status
    locked: bool,

    /// Phase offset in microseconds
    phase_offset_us: i64,

    /// Ring buffer of recent frame arrival times for jitter measurement
    arrival_times: VecDeque<Instant>,

    /// Last measured phase error in microseconds
    last_phase_error_us: f64,

    /// Last measured frequency error in PPM
    last_freq_error_ppm: f64,
}

impl GenlockSync {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            reference_detected: false,
            locked: false,
            phase_offset_us: 0,
            arrival_times: VecDeque::with_capacity(16),
            last_phase_error_us: 0.0,
            last_freq_error_ppm: 0.0,
        }
    }

    /// Record a frame arrival time for variance tracking.
    fn record_arrival(&mut self, now: Instant) {
        self.arrival_times.push_back(now);
        // Keep only the last 16 samples
        while self.arrival_times.len() > 16 {
            self.arrival_times.pop_front();
        }
    }

    /// Measure timing variance across the arrival ring buffer.
    ///
    /// Returns `(phase_error_us, freq_error_ppm)`.
    /// `phase_error_us` is the mean absolute deviation of inter-frame
    /// intervals from the expected frame period.
    /// `freq_error_ppm` is estimated from the linear drift over the window.
    fn measure_variance(&self, expected_frame_period_us: f64) -> (f64, f64) {
        let n = self.arrival_times.len();
        if n < 2 {
            return (0.0, 0.0);
        }

        // Collect inter-frame intervals in microseconds
        let intervals: Vec<f64> = self
            .arrival_times
            .iter()
            .zip(self.arrival_times.iter().skip(1))
            .map(|(a, b)| b.saturating_duration_since(*a).as_micros() as f64)
            .collect();

        // Mean absolute deviation from expected period → phase error proxy
        let mean_dev: f64 = intervals
            .iter()
            .map(|&iv| (iv - expected_frame_period_us).abs())
            .sum::<f64>()
            / intervals.len() as f64;

        // Linear drift: difference between first and last interval as PPM of expected
        let drift_us = intervals
            .last()
            .copied()
            .unwrap_or(expected_frame_period_us)
            - intervals
                .first()
                .copied()
                .unwrap_or(expected_frame_period_us);
        let freq_error_ppm = if expected_frame_period_us > 0.0 {
            (drift_us / expected_frame_period_us) * 1_000_000.0
        } else {
            0.0
        };

        (mean_dev, freq_error_ppm)
    }

    /// Check genlock status.
    ///
    /// Records the current instant as a frame arrival, then evaluates whether
    /// the timing variance is within the 1 µs lock threshold.
    fn check_status(&mut self) -> bool {
        if !self.enabled {
            return true; // Always "locked" when disabled
        }

        let now = Instant::now();
        self.record_arrival(now);
        self.reference_detected = true;

        // Use a nominal 25 fps period as the expected inter-frame interval.
        // In a real system this would come from the reference signal frequency.
        let expected_period_us = 1_000_000.0 / 25.0; // 40 000 µs @ 25 fps
        let (phase_err, freq_err) = self.measure_variance(expected_period_us);

        self.last_phase_error_us = phase_err;
        self.last_freq_error_ppm = freq_err;

        // Lock criterion: mean phase deviation below 1 µs
        self.locked = phase_err < 1.0;

        self.locked
    }

    /// Return the full genlock status.
    #[allow(dead_code)]
    fn genlock_status(&self) -> GenlockStatus {
        GenlockStatus {
            is_locked: self.locked,
            phase_error_us: self.last_phase_error_us,
            frequency_error_ppm: self.last_freq_error_ppm,
        }
    }

    /// Get phase compensation
    #[allow(dead_code)]
    fn phase_compensation(&self) -> i64 {
        if self.locked {
            self.phase_offset_us
        } else {
            0
        }
    }
}

/// Emergency fallback handler
struct FallbackHandler {
    /// Fallback content path
    #[allow(dead_code)]
    fallback_path: PathBuf,

    /// Fallback active
    active: bool,

    /// Fallback reason
    reason: Option<String>,

    /// Activation time
    activation_time: Option<Instant>,
}

impl FallbackHandler {
    fn new(fallback_path: PathBuf) -> Self {
        Self {
            fallback_path,
            active: false,
            reason: None,
            activation_time: None,
        }
    }

    /// Activate fallback
    fn activate(&mut self, reason: String) {
        self.active = true;
        self.reason = Some(reason);
        self.activation_time = Some(Instant::now());
    }

    /// Deactivate fallback
    fn deactivate(&mut self) {
        self.active = false;
        self.reason = None;
        self.activation_time = None;
    }

    /// Check if fallback is active
    fn is_active(&self) -> bool {
        self.active
    }
}

/// Playback engine internal state
struct EngineState {
    /// Current playback state
    state: PlaybackState,

    /// Clock state
    clock: ClockState,

    /// Buffer manager
    buffer: BufferManager,

    /// Genlock synchronizer
    genlock: GenlockSync,

    /// Fallback handler
    fallback: FallbackHandler,

    /// Statistics
    stats: PlaybackStats,

    /// Start time
    start_time: Option<Instant>,
}

/// Real-time playback engine
pub struct PlaybackEngine {
    config: PlaybackConfig,
    state: Arc<RwLock<EngineState>>,
    #[allow(dead_code)]
    frame_tx: Arc<Mutex<Option<mpsc::Sender<FrameBuffer>>>>,
}

impl PlaybackEngine {
    /// Create a new playback engine
    pub fn new(config: PlaybackConfig) -> Result<Self> {
        let buffer_manager = BufferManager::new(config.buffer_size);
        let genlock = GenlockSync::new(config.genlock_enabled);
        let fallback = FallbackHandler::new(config.fallback_path.clone());

        let state = EngineState {
            state: PlaybackState::Stopped,
            clock: ClockState::new(),
            buffer: buffer_manager,
            genlock,
            fallback,
            stats: PlaybackStats::default(),
            start_time: None,
        };

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(state)),
            frame_tx: Arc::new(Mutex::new(None)),
        })
    }

    /// Start playback
    pub async fn start(&self) -> Result<()> {
        let mut state = self.state.write();

        if state.state != PlaybackState::Stopped {
            return Err(PlayoutError::Playback("Engine already running".to_string()));
        }

        state.state = PlaybackState::Buffering;
        state.start_time = Some(Instant::now());
        state.clock = ClockState::new();
        state.stats = PlaybackStats::default();

        Ok(())
    }

    /// Stop playback
    pub async fn stop(&self) -> Result<()> {
        let mut state = self.state.write();
        state.state = PlaybackState::Stopped;
        state.buffer.clear();
        state.fallback.deactivate();

        Ok(())
    }

    /// Pause playback
    pub async fn pause(&self) -> Result<()> {
        let mut state = self.state.write();
        if state.state == PlaybackState::Playing {
            state.state = PlaybackState::Paused;
        }
        Ok(())
    }

    /// Resume playback
    pub async fn resume(&self) -> Result<()> {
        let mut state = self.state.write();
        if state.state == PlaybackState::Paused {
            state.state = PlaybackState::Playing;
        }
        Ok(())
    }

    /// Get current playback state
    pub fn get_state(&self) -> PlaybackState {
        self.state.read().state
    }

    /// Push a frame to the buffer
    pub fn push_frame(&self, frame: FrameBuffer) -> Result<()> {
        let mut state = self.state.write();
        state.buffer.push(frame)?;

        // Transition from buffering to playing when ready
        if state.state == PlaybackState::Buffering && state.buffer.is_ready() {
            state.state = PlaybackState::Playing;
        }

        Ok(())
    }

    /// Get next frame for output
    pub fn get_next_frame(&self) -> Option<FrameBuffer> {
        let mut state = self.state.write();

        if state.state != PlaybackState::Playing {
            return None;
        }

        // Check genlock status
        if !state.genlock.check_status() {
            tracing::warn!("Genlock not locked");
        }

        // Get frame from buffer
        if let Some(mut frame) = state.buffer.pop() {
            // Update statistics
            state.stats.frames_played += 1;
            state.clock.frame_counter += 1;

            // Check for late frames
            let now = Instant::now();
            let expected_time = state.clock.next_frame_time(self.config.video_format.fps());

            if now > expected_time {
                let latency_us = (now - expected_time).as_micros() as u64;
                frame.metadata.flags.late = true;
                state.stats.frames_late += 1;

                if latency_us > state.stats.max_latency_us {
                    state.stats.max_latency_us = latency_us;
                }
            }

            Some(frame)
        } else {
            // Buffer underrun
            state.stats.buffer_underruns += 1;
            state.stats.frames_dropped += 1;

            if state.buffer.level() == 0 {
                state.state = PlaybackState::Buffering;
            }

            // Check if we should activate fallback
            if state.stats.frames_dropped > self.config.max_frame_drops as u64 {
                state
                    .fallback
                    .activate("Too many dropped frames".to_string());
                state.state = PlaybackState::Fallback;
            }

            None
        }
    }

    /// Get playback statistics
    pub fn get_stats(&self) -> PlaybackStats {
        let mut state = self.state.write();

        // Update uptime
        if let Some(start) = state.start_time {
            state.stats.uptime_seconds = start.elapsed().as_secs();
        }

        state.stats.clone()
    }

    /// Get buffer level
    pub fn buffer_level(&self) -> usize {
        self.state.read().buffer.level()
    }

    /// Check if in fallback mode
    pub fn is_fallback_active(&self) -> bool {
        self.state.read().fallback.is_active()
    }

    /// Activate emergency fallback
    pub fn activate_fallback(&self, reason: String) {
        let mut state = self.state.write();
        state.fallback.activate(reason);
        state.state = PlaybackState::Fallback;
    }

    /// Deactivate fallback
    pub fn deactivate_fallback(&self) {
        let mut state = self.state.write();
        state.fallback.deactivate();
        if state.state == PlaybackState::Fallback {
            state.state = PlaybackState::Playing;
        }
    }

    /// Synchronize clock with reference
    pub fn sync_clock(&self, reference: DateTime<Utc>) {
        let mut state = self.state.write();
        state.clock.utc_reference = reference;
        state.clock.last_sync = Instant::now();
        state.clock.reference_time = Instant::now();
    }

    /// Get current frame number
    pub fn current_frame(&self) -> u64 {
        let state = self.state.read();
        state.clock.current_frame(self.config.video_format.fps())
    }

    /// Get current presentation time
    pub fn current_pts(&self) -> DateTime<Utc> {
        let state = self.state.read();
        let elapsed = state.clock.reference_time.elapsed();
        state.clock.utc_reference + Duration::from_std(elapsed).unwrap_or_default()
    }

    /// Wait for next frame time
    pub async fn wait_next_frame(&self) {
        let (next_time, now) = {
            let state = self.state.read();
            let frame_rate = self.config.video_format.fps();
            let next_time = state.clock.next_frame_time(frame_rate);
            let now = Instant::now();
            (next_time, now)
        };

        if next_time > now {
            let wait_duration = next_time - now;
            tokio::time::sleep(wait_duration).await;
        }
    }

    /// Calculate frame timing
    pub fn frame_timing(&self) -> StdDuration {
        let fps = self.config.video_format.fps();
        StdDuration::from_secs_f64(1.0 / fps)
    }

    /// Deterministic presentation time of frame `frame_index`.
    ///
    /// Returns the modeled, wall-clock-free time at which frame `frame_index`
    /// should be presented, measured from the start of playback (frame 0 at
    /// `Duration::ZERO`). This is `frame_index * frame_timing()` adjusted by the
    /// engine's modeled per-frame clock drift (`ClockState::drift_us`, the same
    /// drift applied by the private `ClockState::next_frame_time` scheduler).
    ///
    /// The result is pure and reproducible: it never reads `Instant::now()` and
    /// depends only on the frame index, the configured frame rate, and the
    /// accumulated drift correction. With the default zero drift it yields exact
    /// integer multiples of `frame_timing()`, giving jitter-free, monotonically
    /// increasing presentation timestamps suitable for offline scheduling and
    /// frame-pacing calculations.
    pub fn presentation_time(&self, frame_index: u64) -> StdDuration {
        // Base presentation offset: exact multiple of the frame period. Scale
        // the frame period in nanoseconds using 128-bit arithmetic so the
        // result neither truncates the u64 index nor overflows for large frame
        // counts. Working from `frame_timing()` (rather than truncated integer
        // microseconds) avoids per-frame rounding drift for non-integer frame
        // rates such as 29.97 fps, matching `frame_timing()` exactly on every
        // interval with no accumulated jitter.
        let period_nanos = self.frame_timing().as_nanos();
        let base_nanos = period_nanos.saturating_mul(frame_index as u128);

        // Apply the engine's modeled clock drift, mirroring the per-frame drift
        // term used by `ClockState::next_frame_time`. Drift defaults to zero
        // when no drift input has been supplied via `apply_clock_correction`.
        let drift_us = self.state.read().clock.drift_us;
        let drift_nanos = (drift_us as i128) * 1_000;

        let total_nanos = base_nanos
            .saturating_add_signed(drift_nanos)
            .min(u128::from(u64::MAX));

        // total_nanos is now guaranteed to fit in u64.
        StdDuration::from_nanos(total_nanos as u64)
    }

    /// Check for clock drift
    pub fn check_clock_drift(&self) -> i64 {
        let state = self.state.read();
        state.clock.drift_us
    }

    /// Apply clock correction
    pub fn apply_clock_correction(&self, correction_us: i64) {
        let mut state = self.state.write();
        state.clock.drift_us += correction_us;
        state.stats.clock_drift_us = state.clock.drift_us;
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut state = self.state.write();
        state.stats = PlaybackStats::default();
        state.start_time = Some(Instant::now());
    }

    /// Get genlock status
    pub fn genlock_status(&self) -> (bool, bool) {
        let state = self.state.read();
        (state.genlock.reference_detected, state.genlock.locked)
    }

    /// Create a blank frame (for fill)
    pub fn create_blank_frame(&self, frame_number: u64) -> FrameBuffer {
        let width = self.config.video_format.width();
        let height = self.config.video_format.height();
        let video_size = (width * height * 3) as usize; // RGB24 placeholder

        let audio_samples =
            (self.config.audio_format.sample_rate as f64 / self.config.video_format.fps()) as usize;
        let audio_size = audio_samples * self.config.audio_format.channels as usize * 4; // Float32

        FrameBuffer {
            frame_number,
            pts: self.current_pts(),
            video_data: vec![0; video_size],
            audio_data: vec![0; audio_size],
            width,
            height,
            audio_samples,
            metadata: FrameMetadata::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_playback_config() {
        let playout_config = PlayoutConfig::default();
        let config = PlaybackConfig::from_playout_config(&playout_config);
        assert_eq!(config.buffer_size, 10);
    }

    #[test]
    fn test_clock_state() {
        let clock = ClockState::new();
        let frame = clock.current_frame(25.0);
        assert_eq!(frame, 0);
    }

    #[test]
    fn test_buffer_manager() {
        let buffer = BufferManager::new(10);
        assert_eq!(buffer.level(), 0);
        assert!(!buffer.is_ready());
    }

    #[test]
    fn test_playback_stats() {
        let mut stats = PlaybackStats::default();
        stats.frames_played = 1000;
        stats.frames_dropped = 10;

        assert_eq!(stats.drop_rate(), 1.0);
    }

    #[test]
    fn test_genlock_sync() {
        let mut genlock = GenlockSync::new(false);
        assert!(genlock.check_status()); // Always true when disabled
    }

    #[test]
    fn test_fallback_handler() {
        let mut fallback = FallbackHandler::new(PathBuf::from("/fallback.mxf"));
        assert!(!fallback.is_active());

        fallback.activate("Test".to_string());
        assert!(fallback.is_active());

        fallback.deactivate();
        assert!(!fallback.is_active());
    }

    #[tokio::test]
    async fn test_engine_lifecycle() {
        let config = PlaybackConfig::from_playout_config(&PlayoutConfig::default());
        let engine = PlaybackEngine::new(config).expect("should succeed in test");

        assert_eq!(engine.get_state(), PlaybackState::Stopped);

        engine.start().await.expect("should succeed in test");
        assert_eq!(engine.get_state(), PlaybackState::Buffering);

        engine.stop().await.expect("should succeed in test");
        assert_eq!(engine.get_state(), PlaybackState::Stopped);
    }

    #[test]
    fn test_frame_buffer() {
        let frame = FrameBuffer {
            frame_number: 100,
            pts: Utc::now(),
            video_data: vec![0; 1920 * 1080 * 3],
            audio_data: vec![0; 1920],
            width: 1920,
            height: 1080,
            audio_samples: 1920,
            metadata: FrameMetadata::default(),
        };

        assert_eq!(frame.frame_number, 100);
        assert_eq!(frame.width, 1920);
    }
}
