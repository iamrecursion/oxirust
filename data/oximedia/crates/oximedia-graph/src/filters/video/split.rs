//! Video split (tee/fanout) filter.
//!
//! The [`SplitFilter`] duplicates an incoming video frame to multiple output
//! ports, enabling fan-out topologies where one source feeds several
//! downstream processing branches simultaneously.

#![forbid(unsafe_code)]
#![allow(dead_code)]

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortFormat, PortId, PortType, VideoPortFormat};

/// Configuration for the [`SplitFilter`].
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Number of output ports (fan-out factor).  Must be at least 1.
    pub outputs: usize,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self { outputs: 2 }
    }
}

impl SplitConfig {
    /// Create a split configuration with `n` output ports.
    ///
    /// Clamps `n` to a minimum of 1.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self { outputs: n.max(1) }
    }
}

/// A video filter that duplicates an incoming frame to multiple output ports
/// (tee / fan-out).
///
/// # Ports
///
/// - **Input 0** (`"input"`) – the source video stream.
/// - **Output 0 … N-1** (`"output_0"` … `"output_N-1"`) – one port per fan-out
///   branch.  All outputs receive a clone of the same frame.
///
/// # Example
///
/// ```
/// use oximedia_graph::filters::video::SplitFilter;
/// use oximedia_graph::node::{Node, NodeId};
///
/// let split = SplitFilter::new(NodeId(0), "tee", 3);
/// assert_eq!(split.outputs().len(), 3);
/// ```
pub struct SplitFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    /// Cloned frames waiting to be pulled on each output port.
    ///
    /// The outer `Vec` is indexed by output-port index.  Each inner `Vec` is a
    /// small queue (usually 0 or 1 entries) of frames ready on that port.
    pending: Vec<Vec<FilterFrame>>,
}

impl SplitFilter {
    /// Create a new [`SplitFilter`] with `n` output ports.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, n: usize) -> Self {
        let n = n.max(1);
        let video_format = PortFormat::Video(VideoPortFormat::any());

        let inputs =
            vec![InputPort::new(PortId(0), "input", PortType::Video)
                .with_format(video_format.clone())];

        let outputs: Vec<OutputPort> = (0..n)
            .map(|i| {
                OutputPort::new(PortId(i as u32), format!("output_{i}"), PortType::Video)
                    .with_format(video_format.clone())
            })
            .collect();

        let pending = vec![Vec::new(); n];

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            inputs,
            outputs,
            pending,
        }
    }

    /// Create a [`SplitFilter`] from a [`SplitConfig`].
    #[must_use]
    pub fn from_config(id: NodeId, name: impl Into<String>, config: SplitConfig) -> Self {
        Self::new(id, name, config.outputs)
    }

    /// Number of configured output ports.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Drain one pending frame from the given output port index.
    ///
    /// Returns `None` if no frame is pending on that port.
    pub fn pop_output(&mut self, port_index: usize) -> Option<FilterFrame> {
        self.pending.get_mut(port_index).and_then(|q| {
            if q.is_empty() {
                None
            } else {
                Some(q.remove(0))
            }
        })
    }

    /// Push a frame to all output port queues.
    fn fan_out(&mut self, frame: FilterFrame) {
        let n = self.pending.len();
        for i in 0..n {
            // The last port can take ownership; all prior ports get a clone.
            if i + 1 < n {
                self.pending[i].push(frame.clone());
            } else {
                self.pending[i].push(frame.clone());
            }
        }
    }
}

impl Node for SplitFilter {
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

    /// Process an incoming video frame by fanning it out to all output port
    /// queues.
    ///
    /// Returns the frame on **output port 0** immediately so that the single
    /// `process` return value is usable by the first downstream node.  Frames
    /// for output ports 1..N are stored in the internal pending queues and can
    /// be retrieved via [`SplitFilter::pop_output`].
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
                // Fan out to all queues.
                self.fan_out(frame);

                // Return port-0 frame immediately; others stay in queue.
                Ok(self.pending[0].pop())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_codec::VideoFrame;
    use oximedia_core::PixelFormat;

    fn make_video_frame() -> FilterFrame {
        FilterFrame::Video(VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080))
    }

    #[test]
    fn test_split_creation_default_two_outputs() {
        let split = SplitFilter::new(NodeId(0), "tee", 2);
        assert_eq!(split.output_count(), 2);
        assert_eq!(split.outputs().len(), 2);
        assert_eq!(split.inputs().len(), 1);
    }

    #[test]
    fn test_split_creation_n_outputs() {
        let split = SplitFilter::new(NodeId(1), "tee4", 4);
        assert_eq!(split.output_count(), 4);
    }

    #[test]
    fn test_split_clamps_to_minimum_one() {
        let split = SplitFilter::new(NodeId(2), "tee_min", 0);
        assert_eq!(split.output_count(), 1);
    }

    #[test]
    fn test_split_from_config() {
        let config = SplitConfig::new(3);
        let split = SplitFilter::from_config(NodeId(0), "cfg_tee", config);
        assert_eq!(split.output_count(), 3);
    }

    #[test]
    fn test_split_process_returns_port0_frame() {
        let mut split = SplitFilter::new(NodeId(0), "tee", 2);
        let frame = make_video_frame();
        let result = split.process(Some(frame)).expect("process should succeed");
        assert!(result.is_some());
        assert!(result.expect("value should exist").is_video());
    }

    #[test]
    fn test_split_process_none_returns_none() {
        let mut split = SplitFilter::new(NodeId(0), "tee", 2);
        let result = split.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_split_pending_on_additional_outputs() {
        let mut split = SplitFilter::new(NodeId(0), "tee", 3);
        let frame = make_video_frame();
        split.process(Some(frame)).expect("process should succeed");
        // Port 1 and port 2 should have a pending frame each.
        assert!(split.pop_output(1).is_some());
        assert!(split.pop_output(2).is_some());
        // No more frames pending.
        assert!(split.pop_output(1).is_none());
    }

    #[test]
    fn test_split_port_names() {
        let split = SplitFilter::new(NodeId(0), "tee", 3);
        assert_eq!(split.outputs()[0].name, "output_0");
        assert_eq!(split.outputs()[1].name, "output_1");
        assert_eq!(split.outputs()[2].name, "output_2");
    }

    #[test]
    fn test_split_node_type_is_filter() {
        let split = SplitFilter::new(NodeId(0), "tee", 2);
        assert_eq!(split.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_split_state_transitions() {
        let mut split = SplitFilter::new(NodeId(0), "tee", 2);
        assert_eq!(split.state(), NodeState::Idle);
        split
            .set_state(NodeState::Processing)
            .expect("state transition should succeed");
        assert_eq!(split.state(), NodeState::Processing);
    }

    #[test]
    fn test_split_audio_frame_returns_error() {
        use oximedia_audio::{AudioFrame, ChannelLayout};
        use oximedia_core::SampleFormat;
        let mut split = SplitFilter::new(NodeId(0), "tee", 2);
        let audio_frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Stereo);
        let result = split.process(Some(FilterFrame::Audio(audio_frame)));
        assert!(result.is_err());
    }

    #[test]
    fn test_split_config_default() {
        let config = SplitConfig::default();
        assert_eq!(config.outputs, 2);
    }

    #[test]
    fn test_split_multiple_frames_queued() {
        let mut split = SplitFilter::new(NodeId(0), "tee", 2);
        for _ in 0..3 {
            let frame = make_video_frame();
            split.process(Some(frame)).expect("process should succeed");
        }
        // Port 1 should have 3 pending frames.
        assert!(split.pop_output(1).is_some());
        assert!(split.pop_output(1).is_some());
        assert!(split.pop_output(1).is_some());
        assert!(split.pop_output(1).is_none());
    }
}
