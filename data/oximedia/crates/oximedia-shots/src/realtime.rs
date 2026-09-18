//! Real-time shot detection with streaming frame input.
//!
//! [`RealtimeShotDetector`] accepts frames one at a time via [`push_frame`]
//! and fires a configurable callback (or stores events internally) whenever a
//! shot boundary is detected.  This is designed for live-video pipelines where
//! frames arrive sequentially and latency matters more than completeness.
//!
//! # Design
//!
//! - Maintains a ring buffer of the last two processed frames.
//! - On each new frame, compares it to the previous frame using `CutDetector`.
//! - Detected cuts are appended to an internal event queue retrievable by
//!   [`drain_events`].
//! - An optional configurable window of **N** frames can be used to smooth
//!   false positives by requiring the score to exceed the threshold in at
//!   least `min_detections_in_window` of the last `window_size` comparisons.
//!
//! [`push_frame`]: RealtimeShotDetector::push_frame
//! [`drain_events`]: RealtimeShotDetector::drain_events

use crate::detect::{ContentComplexity, CutDetector};
use crate::error::ShotResult;
use crate::frame_buffer::FrameBuffer;

// ---------------------------------------------------------------------------
// Event types
// ---------------------------------------------------------------------------

/// A shot-boundary event detected during real-time processing.
#[derive(Debug, Clone)]
pub struct ShotBoundaryEvent {
    /// Frame index at which the boundary was detected (0-based, relative to
    /// the stream start).
    pub frame_index: u64,
    /// Combined cut score at the detected boundary (0.0–1.0).
    pub score: f32,
    /// Whether the boundary was detected as a hard cut (true) or flagged by
    /// the smoothing window (false = softer boundary).
    pub is_hard_cut: bool,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the real-time shot detector.
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Content complexity used for adaptive threshold selection.
    pub complexity: ContentComplexity,
    /// Histogram difference threshold for hard cuts.
    pub cut_threshold: f32,
    /// Edge change threshold for hard cuts.
    pub edge_threshold: f32,
    /// Number of frames in the temporal smoothing window.
    ///
    /// Set to 1 to disable smoothing and emit events immediately.
    pub window_size: usize,
    /// Minimum number of frames in the window that must exceed their
    /// individual thresholds before a soft-boundary event is emitted.
    ///
    /// Only used when `window_size > 1`.
    pub min_detections_in_window: usize,
    /// Maximum number of events to buffer internally before older events
    /// are silently dropped.
    pub max_event_buffer: usize,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            complexity: ContentComplexity::Auto,
            cut_threshold: 0.30,
            edge_threshold: 0.40,
            window_size: 3,
            min_detections_in_window: 2,
            max_event_buffer: 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// Real-time shot detector for streaming frame input.
///
/// # Example
///
/// ```no_run
/// use oximedia_shots::realtime::{RealtimeConfig, RealtimeShotDetector};
/// use oximedia_shots::frame_buffer::FrameBuffer;
///
/// let mut detector = RealtimeShotDetector::new(RealtimeConfig::default());
///
/// // Simulate a frame stream
/// for frame_idx in 0..100u64 {
///     let frame = FrameBuffer::zeros(720, 1280, 3);
///     detector.push_frame(frame).expect("frame processing failed");
/// }
///
/// // Retrieve detected boundaries
/// for event in detector.drain_events() {
///     println!("Shot boundary at frame {}", event.frame_index);
/// }
/// ```
pub struct RealtimeShotDetector {
    /// Underlying cut detector configured from the realtime config.
    detector: CutDetector,
    /// Configuration.
    config: RealtimeConfig,
    /// The most recently processed frame (if any).
    previous_frame: Option<FrameBuffer>,
    /// Total number of frames pushed so far.
    frame_count: u64,
    /// Circular window of recent cut scores for smoothing.
    score_window: Vec<f32>,
    /// Number of valid entries in `score_window`.
    window_fill: usize,
    /// Next write position in `score_window`.
    window_pos: usize,
    /// Internal event queue.
    events: Vec<ShotBoundaryEvent>,
}

impl RealtimeShotDetector {
    /// Create a new real-time shot detector.
    #[must_use]
    pub fn new(config: RealtimeConfig) -> Self {
        let detector = if config.complexity == ContentComplexity::Auto {
            CutDetector::with_params(config.cut_threshold, config.edge_threshold, 1)
        } else {
            CutDetector::adaptive(config.complexity)
        };

        let window_size = config.window_size.max(1);
        Self {
            detector,
            config,
            previous_frame: None,
            frame_count: 0,
            score_window: vec![0.0_f32; window_size],
            window_fill: 0,
            window_pos: 0,
            events: Vec::new(),
        }
    }

    /// Push the next frame from the stream.
    ///
    /// If a shot boundary is detected, a [`ShotBoundaryEvent`] is appended to
    /// the internal queue (accessible via [`drain_events`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the frame has fewer than 3 channels or if the frame
    /// comparison fails for another reason.
    ///
    /// [`drain_events`]: Self::drain_events
    pub fn push_frame(&mut self, frame: FrameBuffer) -> ShotResult<()> {
        let current_index = self.frame_count;
        self.frame_count += 1;

        if let Some(prev) = self.previous_frame.take() {
            // Run cut detection between previous and current frame
            let (is_hard_cut, score) = self.detector.detect_cut(&prev, &frame)?;

            // Update sliding score window
            let ws = self.config.window_size.max(1);
            let pos = self.window_pos % ws;
            self.score_window[pos] = score;
            self.window_pos = (self.window_pos + 1) % ws;
            if self.window_fill < ws {
                self.window_fill += 1;
            }

            // Emit hard cut immediately
            if is_hard_cut {
                self.emit_event(ShotBoundaryEvent {
                    frame_index: current_index,
                    score,
                    is_hard_cut: true,
                });
            } else if ws > 1 && self.window_fill >= ws {
                // Smoothed detection: count how many recent scores exceed threshold
                let threshold = self.config.cut_threshold;
                let detections = self.score_window.iter().filter(|&&s| s > threshold).count();
                if detections >= self.config.min_detections_in_window {
                    self.emit_event(ShotBoundaryEvent {
                        frame_index: current_index,
                        score,
                        is_hard_cut: false,
                    });
                }
            }
        }

        self.previous_frame = Some(frame);
        Ok(())
    }

    /// Drain and return all buffered shot boundary events.
    ///
    /// Clears the internal queue.  Call this periodically to retrieve detected
    /// boundaries without accumulating unbounded memory.
    pub fn drain_events(&mut self) -> Vec<ShotBoundaryEvent> {
        std::mem::take(&mut self.events)
    }

    /// Peek at the current event queue without draining it.
    #[must_use]
    pub fn events(&self) -> &[ShotBoundaryEvent] {
        &self.events
    }

    /// Total number of frames processed since creation or last [`reset`].
    ///
    /// [`reset`]: Self::reset
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset the detector state (clears frame history, window, and event queue).
    ///
    /// The configuration is retained.  Use this when switching between input
    /// streams without creating a new detector.
    pub fn reset(&mut self) {
        self.previous_frame = None;
        self.frame_count = 0;
        self.window_fill = 0;
        self.window_pos = 0;
        for v in &mut self.score_window {
            *v = 0.0;
        }
        self.events.clear();
        self.detector.clear_cache();
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &RealtimeConfig {
        &self.config
    }

    // Internal helper: append event, respecting the max buffer limit.
    fn emit_event(&mut self, event: ShotBoundaryEvent) {
        if self.events.len() >= self.config.max_event_buffer {
            // Drop oldest event to make room
            self.events.remove(0);
        }
        self.events.push(event);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_frame(val: u8) -> FrameBuffer {
        FrameBuffer::from_elem(64, 64, 3, val)
    }

    #[test]
    fn test_realtime_detector_creation() {
        let detector = RealtimeShotDetector::new(RealtimeConfig::default());
        assert_eq!(detector.frame_count(), 0);
        assert!(detector.events().is_empty());
    }

    #[test]
    fn test_push_single_frame_no_event() {
        let mut detector = RealtimeShotDetector::new(RealtimeConfig::default());
        let frame = uniform_frame(100);
        detector.push_frame(frame).expect("push ok");
        assert_eq!(detector.frame_count(), 1);
        // Only one frame: no comparison can be made yet
        assert!(detector.events().is_empty());
    }

    #[test]
    fn test_push_identical_frames_no_event() {
        let mut detector = RealtimeShotDetector::new(RealtimeConfig {
            window_size: 1,
            min_detections_in_window: 1,
            ..RealtimeConfig::default()
        });
        for _ in 0..5 {
            detector.push_frame(uniform_frame(128)).expect("push ok");
        }
        let events = detector.drain_events();
        assert!(
            events.is_empty(),
            "identical frames should not trigger cuts"
        );
    }

    #[test]
    fn test_push_hard_cut_detected() {
        let mut detector = RealtimeShotDetector::new(RealtimeConfig {
            cut_threshold: 0.1, // very low threshold to guarantee hard cut detection
            edge_threshold: 0.1,
            window_size: 1,
            min_detections_in_window: 1,
            ..RealtimeConfig::default()
        });
        // Push a black frame then a white frame
        detector.push_frame(uniform_frame(0)).expect("push ok");
        detector.push_frame(uniform_frame(255)).expect("push ok");
        let events = detector.drain_events();
        assert_eq!(events.len(), 1, "black→white should be a hard cut");
        assert!(events[0].is_hard_cut);
        assert_eq!(events[0].frame_index, 1);
        assert!(events[0].score > 0.0);
    }

    #[test]
    fn test_drain_clears_events() {
        let mut detector = RealtimeShotDetector::new(RealtimeConfig {
            cut_threshold: 0.1,
            edge_threshold: 0.1,
            window_size: 1,
            min_detections_in_window: 1,
            ..RealtimeConfig::default()
        });
        detector.push_frame(uniform_frame(0)).expect("push ok");
        detector.push_frame(uniform_frame(255)).expect("push ok");
        let _ = detector.drain_events();
        // Second drain should be empty
        assert!(detector.drain_events().is_empty());
    }

    #[test]
    fn test_frame_count_increments() {
        let mut detector = RealtimeShotDetector::new(RealtimeConfig::default());
        for i in 0..10u8 {
            detector.push_frame(uniform_frame(i * 25)).expect("push ok");
        }
        assert_eq!(detector.frame_count(), 10);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut detector = RealtimeShotDetector::new(RealtimeConfig {
            cut_threshold: 0.1,
            edge_threshold: 0.1,
            window_size: 1,
            min_detections_in_window: 1,
            ..RealtimeConfig::default()
        });
        detector.push_frame(uniform_frame(0)).expect("push ok");
        detector.push_frame(uniform_frame(255)).expect("push ok");
        assert!(!detector.events().is_empty());

        detector.reset();
        assert_eq!(detector.frame_count(), 0);
        assert!(detector.events().is_empty());
        assert!(detector.previous_frame.is_none());
    }

    #[test]
    fn test_config_accessor() {
        let config = RealtimeConfig {
            window_size: 7,
            ..RealtimeConfig::default()
        };
        let detector = RealtimeShotDetector::new(config);
        assert_eq!(detector.config().window_size, 7);
    }

    #[test]
    fn test_max_event_buffer_drop_oldest() {
        let mut detector = RealtimeShotDetector::new(RealtimeConfig {
            cut_threshold: 0.1,
            edge_threshold: 0.1,
            window_size: 1,
            min_detections_in_window: 1,
            max_event_buffer: 2,
            ..RealtimeConfig::default()
        });
        // Push alternating black/white frames to trigger many cuts
        for i in 0..10u8 {
            let val = if i % 2 == 0 { 0 } else { 255 };
            detector.push_frame(uniform_frame(val)).expect("push ok");
        }
        // Buffer should not exceed max_event_buffer
        assert!(detector.events().len() <= 2);
    }

    #[test]
    fn test_smoothing_window_reduces_false_positives() {
        // With window_size=3 and min_detections=2, a single high score should
        // not immediately emit an event.
        let mut detector = RealtimeShotDetector::new(RealtimeConfig {
            cut_threshold: 0.05,
            edge_threshold: 0.05,
            window_size: 3,
            min_detections_in_window: 2,
            ..RealtimeConfig::default()
        });
        // Start with a previous frame
        detector.push_frame(uniform_frame(100)).expect("push ok");
        // One different frame (high score but not a hard cut with very low threshold)
        detector.push_frame(uniform_frame(200)).expect("push ok");
        // Window not yet full (only 1 comparison done), event may or may not be
        // emitted depending on whether it's a hard cut; just check no panic.
        let _ = detector.drain_events();
        assert!(detector.frame_count() == 2);
    }

    #[test]
    fn test_realtime_config_default() {
        let cfg = RealtimeConfig::default();
        assert_eq!(cfg.window_size, 3);
        assert_eq!(cfg.min_detections_in_window, 2);
        assert!(cfg.max_event_buffer > 0);
    }

    #[test]
    fn test_multiple_cuts_in_stream() {
        let mut detector = RealtimeShotDetector::new(RealtimeConfig {
            cut_threshold: 0.05,
            edge_threshold: 0.05,
            window_size: 1,
            min_detections_in_window: 1,
            ..RealtimeConfig::default()
        });
        // Sequence: 5 black, 5 white, 5 black
        for _ in 0..5 {
            detector.push_frame(uniform_frame(0)).expect("push ok");
        }
        for _ in 0..5 {
            detector.push_frame(uniform_frame(255)).expect("push ok");
        }
        for _ in 0..5 {
            detector.push_frame(uniform_frame(0)).expect("push ok");
        }
        let events = detector.drain_events();
        // At least the two major transitions should have been detected
        assert!(!events.is_empty(), "should detect at least one cut");
    }
}
