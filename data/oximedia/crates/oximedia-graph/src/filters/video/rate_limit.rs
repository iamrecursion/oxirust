//! Rate-limiting video filter.
//!
//! The [`RateLimitFilter`] throttles video frame throughput to maintain a target
//! frame rate by dropping frames whose presentation timestamps arrive faster than
//! the target cadence.  This is useful for real-time playback and live preview
//! pipelines where the downstream sink (display, encoder) can only consume frames
//! at a fixed pace.
//!
//! # Algorithm
//!
//! For each incoming frame the filter computes the *expected* PTS for the next
//! allowed frame using the target frame duration (`1 / target_fps` seconds,
//! expressed in the frame's timebase).  Frames whose PTS is earlier than the
//! threshold are dropped; frames at or after the threshold are passed through and
//! the threshold is advanced by one frame duration.
//!
//! The filter is stateful: it remembers the PTS of the last emitted frame so that
//! the decisions remain consistent across calls.
//!
//! # Example
//!
//! ```
//! use oximedia_graph::filters::video::RateLimitFilter;
//! use oximedia_graph::node::{Node, NodeId};
//!
//! // Limit output to 24 fps
//! let filter = RateLimitFilter::new(NodeId(0), "rate_limit", 24.0);
//! assert_eq!(filter.node_type(), oximedia_graph::node::NodeType::Filter);
//! ```

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortFormat, PortId, PortType, VideoPortFormat};

/// Configuration for the [`RateLimitFilter`].
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Target output frame rate in frames per second.  Must be positive.
    pub target_fps: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self { target_fps: 30.0 }
    }
}

impl RateLimitConfig {
    /// Create a configuration targeting `fps` frames per second.
    ///
    /// Clamps `fps` to a minimum of `f64::MIN_POSITIVE`.
    #[must_use]
    pub fn new(fps: f64) -> Self {
        Self {
            target_fps: fps.max(f64::MIN_POSITIVE),
        }
    }
}

/// A video filter that limits frame throughput to a target frame rate.
///
/// Frames that arrive too early relative to the last-emitted frame are
/// **dropped**.  The filter does **not** duplicate or interpolate frames; it
/// only acts as a gate.
///
/// # Ports
///
/// - **Input 0** (`"input"`) – incoming video stream.
/// - **Output 0** (`"output"`) – rate-limited video stream.
pub struct RateLimitFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    /// Target frames per second.
    target_fps: f64,
    /// PTS of the last frame that was emitted.  `None` means no frame has been
    /// emitted yet (first frame always passes).
    last_emit_pts: Option<u64>,
    /// Running count of frames dropped since last reset.
    frames_dropped: u64,
    /// Running count of frames emitted since last reset.
    frames_emitted: u64,
}

impl RateLimitFilter {
    /// Create a new [`RateLimitFilter`] with the given target FPS.
    ///
    /// `target_fps` is clamped to a minimum of `f64::MIN_POSITIVE`.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, target_fps: f64) -> Self {
        let video_format = PortFormat::Video(VideoPortFormat::any());
        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Video)
                .with_format(video_format.clone())],
            outputs: vec![
                OutputPort::new(PortId(0), "output", PortType::Video).with_format(video_format)
            ],
            target_fps: target_fps.max(f64::MIN_POSITIVE),
            last_emit_pts: None,
            frames_dropped: 0,
            frames_emitted: 0,
        }
    }

    /// Create a [`RateLimitFilter`] from a [`RateLimitConfig`].
    #[must_use]
    pub fn from_config(id: NodeId, name: impl Into<String>, config: RateLimitConfig) -> Self {
        Self::new(id, name, config.target_fps)
    }

    /// Return the configured target frame rate.
    #[must_use]
    pub fn target_fps(&self) -> f64 {
        self.target_fps
    }

    /// Return the number of frames dropped since the last reset.
    #[must_use]
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped
    }

    /// Return the number of frames emitted since the last reset.
    #[must_use]
    pub fn frames_emitted(&self) -> u64 {
        self.frames_emitted
    }

    /// Decide whether a frame with the given PTS should be emitted.
    ///
    /// The PTS unit is assumed to be the raw integer value stored in the frame's
    /// [`Timestamp`][oximedia_core::Timestamp].  The method advances the internal
    /// state when `true` is returned.
    ///
    /// The first frame (when no prior frame has been emitted) always passes.
    ///
    /// # Parameters
    ///
    /// - `frame_pts` – raw PTS of the candidate frame.
    /// - `timebase_den` – denominator of the stream timebase (e.g. `1000` for
    ///   millisecond PTS, `90000` for MPEG-TS).  Used to convert the target FPS
    ///   into PTS ticks.  Pass `0` to use a default of `90000`.
    pub fn should_emit(&mut self, frame_pts: u64, timebase_den: u64) -> bool {
        let den = if timebase_den == 0 {
            90_000
        } else {
            timebase_den
        };
        // Frame duration in PTS ticks: den / target_fps
        let frame_duration_ticks = (den as f64 / self.target_fps).round() as u64;

        match self.last_emit_pts {
            None => {
                // Very first frame — always emit.
                self.last_emit_pts = Some(frame_pts);
                self.frames_emitted = self.frames_emitted.saturating_add(1);
                true
            }
            Some(last_pts) => {
                let next_threshold = last_pts.saturating_add(frame_duration_ticks);
                if frame_pts >= next_threshold {
                    self.last_emit_pts = Some(frame_pts);
                    self.frames_emitted = self.frames_emitted.saturating_add(1);
                    true
                } else {
                    self.frames_dropped = self.frames_dropped.saturating_add(1);
                    false
                }
            }
        }
    }

    /// Reset all internal state (last emitted PTS, counters).
    pub fn reset_state(&mut self) {
        self.last_emit_pts = None;
        self.frames_dropped = 0;
        self.frames_emitted = 0;
    }
}

impl Node for RateLimitFilter {
    fn id(&self) -> NodeId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn node_type(&self) -> NodeType {
        NodeType::Filter
    }

    fn state(&self) -> NodeState {
        self.state
    }

    fn set_state(&mut self, state: NodeState) -> GraphResult<()> {
        if !self.state.can_transition_to(state) {
            return Err(GraphError::InvalidStateTransition {
                node: self.id,
                from: self.state.to_string(),
                to: state.to_string(),
            });
        }
        self.state = state;
        Ok(())
    }

    fn inputs(&self) -> &[InputPort] {
        &self.inputs
    }

    fn outputs(&self) -> &[OutputPort] {
        &self.outputs
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.reset_state();
        self.set_state(NodeState::Idle)
    }

    /// Process one frame, dropping it if it arrives ahead of the target cadence.
    ///
    /// The timebase denominator is read from the frame's
    /// [`Timestamp`][oximedia_core::Timestamp] rational; if the timebase is zero
    /// or unavailable a default of `90000` is used.
    fn process(&mut self, input: Option<FilterFrame>) -> GraphResult<Option<FilterFrame>> {
        match input {
            None => Ok(None),
            Some(frame) => {
                if !frame.is_video() {
                    return Err(GraphError::PortTypeMismatch {
                        expected: "Video".to_string(),
                        actual: "Audio".to_string(),
                    });
                }

                let ts = frame.timestamp();
                let pts = if ts.pts < 0 { 0u64 } else { ts.pts as u64 };
                // Derive timebase denominator from the frame's rational.
                let timebase_den = ts.timebase.den as u64;

                if self.should_emit(pts, timebase_den) {
                    Ok(Some(frame))
                } else {
                    // Frame dropped — return None to signal "no output this cycle".
                    Ok(None)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_codec::VideoFrame;
    use oximedia_core::{PixelFormat, Rational, Timestamp};

    fn make_video_frame(pts: i64) -> FilterFrame {
        let mut vf = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
        vf.timestamp = Timestamp::new(pts, Rational::new(1, 90_000));
        FilterFrame::Video(vf)
    }

    #[test]
    fn test_rate_limit_creation() {
        let f = RateLimitFilter::new(NodeId(0), "rl", 30.0);
        assert_eq!(f.id(), NodeId(0));
        assert_eq!(f.name(), "rl");
        assert_eq!(f.node_type(), NodeType::Filter);
        assert_eq!(f.state(), NodeState::Idle);
        assert!((f.target_fps() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rate_limit_from_config() {
        let cfg = RateLimitConfig::new(24.0);
        let f = RateLimitFilter::from_config(NodeId(1), "rl24", cfg);
        assert!((f.target_fps() - 24.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rate_limit_config_default() {
        let cfg = RateLimitConfig::default();
        assert!((cfg.target_fps - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_first_frame_always_emits() {
        let mut f = RateLimitFilter::new(NodeId(0), "rl", 30.0);
        // First frame should always pass regardless of PTS.
        let frame = make_video_frame(0);
        let result = f.process(Some(frame)).expect("process should succeed");
        assert!(result.is_some());
        assert_eq!(f.frames_emitted(), 1);
        assert_eq!(f.frames_dropped(), 0);
    }

    #[test]
    fn test_frame_drop_within_duration() {
        // 30 fps at 90_000 timebase → frame duration = 3000 ticks.
        let mut f = RateLimitFilter::new(NodeId(0), "rl", 30.0);
        // Frame 0 passes.
        f.process(Some(make_video_frame(0)))
            .expect("process should succeed");
        // Frame at pts=1000 < 3000 → should be dropped.
        let result = f
            .process(Some(make_video_frame(1000)))
            .expect("process should succeed");
        assert!(result.is_none(), "early frame should be dropped");
        assert_eq!(f.frames_dropped(), 1);
    }

    #[test]
    fn test_frame_passes_after_duration() {
        // 30 fps at 90_000 → frame duration = 3000 ticks.
        let mut f = RateLimitFilter::new(NodeId(0), "rl", 30.0);
        f.process(Some(make_video_frame(0)))
            .expect("process should succeed");
        // Frame at pts=3000 → exactly at boundary → should pass.
        let result = f
            .process(Some(make_video_frame(3000)))
            .expect("process should succeed");
        assert!(result.is_some(), "frame at boundary should pass");
        assert_eq!(f.frames_emitted(), 2);
    }

    #[test]
    fn test_none_input_returns_none() {
        let mut f = RateLimitFilter::new(NodeId(0), "rl", 30.0);
        let result = f.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_audio_frame_returns_error() {
        use oximedia_audio::{AudioFrame, ChannelLayout};
        use oximedia_core::SampleFormat;
        let mut f = RateLimitFilter::new(NodeId(0), "rl", 30.0);
        let audio = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Stereo);
        let result = f.process(Some(FilterFrame::Audio(audio)));
        assert!(result.is_err());
    }

    #[test]
    fn test_reset_clears_state() {
        let mut f = RateLimitFilter::new(NodeId(0), "rl", 30.0);
        f.process(Some(make_video_frame(0)))
            .expect("process should succeed");
        f.reset_state();
        assert_eq!(f.frames_emitted(), 0);
        assert_eq!(f.frames_dropped(), 0);
        // After reset, first frame should pass again.
        let result = f
            .process(Some(make_video_frame(500)))
            .expect("process should succeed");
        assert!(result.is_some());
    }

    #[test]
    fn test_should_emit_direct() {
        let mut f = RateLimitFilter::new(NodeId(0), "rl", 30.0);
        // First call — always true.
        assert!(f.should_emit(0, 90_000));
        // Next threshold = 0 + 3000 = 3000.
        assert!(!f.should_emit(1000, 90_000));
        assert!(!f.should_emit(2999, 90_000));
        assert!(f.should_emit(3000, 90_000));
    }

    #[test]
    fn test_state_transitions() {
        let mut f = RateLimitFilter::new(NodeId(0), "rl", 30.0);
        assert!(f.set_state(NodeState::Processing).is_ok());
        assert_eq!(f.state(), NodeState::Processing);
        assert!(f.set_state(NodeState::Idle).is_ok());
    }

    #[test]
    fn test_node_ports() {
        let f = RateLimitFilter::new(NodeId(0), "rl", 30.0);
        assert_eq!(f.inputs().len(), 1);
        assert_eq!(f.outputs().len(), 1);
        assert_eq!(f.inputs()[0].port_type, PortType::Video);
        assert_eq!(f.outputs()[0].port_type, PortType::Video);
    }
}
