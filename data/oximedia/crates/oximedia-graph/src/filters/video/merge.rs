//! Video merge (picture-in-picture / multi-input compositor) filter.
//!
//! The [`MergeFilter`] combines multiple input video streams into a single
//! output frame.  Each input stream is placed at a configurable position and
//! optionally scaled to a given size, creating picture-in-picture, split-screen
//! or tile-grid compositions.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortFormat, PortId, PortType, VideoPortFormat};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

// ─────────────────────────────────────────────────────────────────────────────
// Placement
// ─────────────────────────────────────────────────────────────────────────────

/// Placement and optional resize specification for one input slot.
#[derive(Debug, Clone)]
pub struct InputPlacement {
    /// X offset in the output frame (pixels, top-left origin).
    pub x: u32,
    /// Y offset in the output frame (pixels, top-left origin).
    pub y: u32,
    /// Optional width override.  `None` keeps the source width.
    pub width: Option<u32>,
    /// Optional height override.  `None` keeps the source height.
    pub height: Option<u32>,
    /// Blend alpha: 0.0 (transparent) … 1.0 (opaque).
    pub alpha: f32,
}

impl Default for InputPlacement {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            width: None,
            height: None,
            alpha: 1.0,
        }
    }
}

impl InputPlacement {
    /// Create a placement at the given position with no resize and full opacity.
    #[must_use]
    pub fn at(x: u32, y: u32) -> Self {
        Self {
            x,
            y,
            ..Default::default()
        }
    }

    /// Set width/height override.
    #[must_use]
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Set the blend alpha.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MergeConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the [`MergeFilter`].
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Width of the output frame.
    pub output_width: u32,
    /// Height of the output frame.
    pub output_height: u32,
    /// Pixel format for the output frame.
    pub output_format: PixelFormat,
    /// Placement spec for each input (index corresponds to input port).
    pub placements: Vec<InputPlacement>,
}

impl MergeConfig {
    /// Create a merge config for `n` inputs into a canvas of the given size.
    ///
    /// Inputs are tiled horizontally by default with equal widths.
    #[must_use]
    pub fn tiled(n: usize, output_width: u32, output_height: u32) -> Self {
        let n = n.max(1);
        let tile_w = output_width / n as u32;
        let placements = (0..n)
            .map(|i| InputPlacement::at(i as u32 * tile_w, 0).with_size(tile_w, output_height))
            .collect();
        Self {
            output_width,
            output_height,
            output_format: PixelFormat::Yuv420p,
            placements,
        }
    }

    /// Create a picture-in-picture config.
    ///
    /// The background is the first input; the overlay is the second input,
    /// placed at `pip_x, pip_y` with a size of `pip_w × pip_h`.
    #[must_use]
    pub fn picture_in_picture(
        bg_width: u32,
        bg_height: u32,
        pip_x: u32,
        pip_y: u32,
        pip_w: u32,
        pip_h: u32,
    ) -> Self {
        Self {
            output_width: bg_width,
            output_height: bg_height,
            output_format: PixelFormat::Yuv420p,
            placements: vec![
                InputPlacement::at(0, 0).with_size(bg_width, bg_height),
                InputPlacement::at(pip_x, pip_y).with_size(pip_w, pip_h),
            ],
        }
    }

    /// Number of configured input slots.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.placements.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MergeFilter
// ─────────────────────────────────────────────────────────────────────────────

/// A video filter that composites multiple input streams into a single output
/// frame.
///
/// # Ports
///
/// - **Input 0 … N-1** (`"input_0"` … `"input_N-1"`) – one port per input
///   stream.
/// - **Output 0** (`"output"`) – the composited output frame.
///
/// # Buffering model
///
/// Each input slot has an internal queue.  Frames are pushed with
/// [`MergeFilter::push_input`] and a composite output is generated via
/// [`Node::process`] once all slots have at least one frame buffered.
pub struct MergeFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    config: MergeConfig,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    /// Per-slot frame queues.  Index == input port index.
    input_queues: Vec<Vec<FilterFrame>>,
}

impl MergeFilter {
    /// Create a new [`MergeFilter`] from a [`MergeConfig`].
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: MergeConfig) -> Self {
        let n = config.input_count().max(1);
        let video_format = PortFormat::Video(VideoPortFormat::any());

        let inputs: Vec<InputPort> = (0..n)
            .map(|i| {
                InputPort::new(PortId(i as u32), format!("input_{i}"), PortType::Video)
                    .with_format(video_format.clone())
            })
            .collect();

        let outputs =
            vec![OutputPort::new(PortId(0), "output", PortType::Video).with_format(video_format)];

        let input_queues = vec![Vec::new(); n];

        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            config,
            inputs,
            outputs,
            input_queues,
        }
    }

    /// Push a frame into the queue for input slot `port_index`.
    ///
    /// # Errors
    ///
    /// Returns `Err` when `port_index` is out of range or the frame is not a
    /// video frame.
    pub fn push_input(&mut self, port_index: usize, frame: FilterFrame) -> GraphResult<()> {
        if port_index >= self.input_queues.len() {
            return Err(GraphError::PortNotFound {
                node: self.id,
                port: PortId(port_index as u32),
            });
        }
        if !frame.is_video() {
            return Err(GraphError::PortTypeMismatch {
                expected: "Video".to_string(),
                actual: "Audio".to_string(),
            });
        }
        self.input_queues[port_index].push(frame);
        Ok(())
    }

    /// Return `true` when every input slot has at least one buffered frame and
    /// a composite output is ready to be produced.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.input_queues.iter().all(|q| !q.is_empty())
    }

    /// Number of configured input ports.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.config.input_count()
    }

    /// Composite all buffered inputs into a single output frame using a simple
    /// nearest-neighbour blit.
    ///
    /// Each input frame is blitted into the output canvas at the position
    /// specified by [`InputPlacement`].  Y-plane blending uses integer alpha
    /// mixing; UV planes are blitted without alpha for simplicity.
    fn composite(&mut self) -> GraphResult<FilterFrame> {
        let out_w = self.config.output_width as usize;
        let out_h = self.config.output_height as usize;

        // Create output frame (Yuv420p: Y = W*H, U = W/2*H/2, V = W/2*H/2).
        let y_size = out_w * out_h;
        let uv_size = (out_w / 2) * (out_h / 2);
        let mut y_plane = vec![0u8; y_size];
        let mut u_plane = vec![128u8; uv_size]; // neutral chroma
        let mut v_plane = vec![128u8; uv_size];

        let n = self.input_queues.len();
        for slot in 0..n {
            let frame = match self.input_queues[slot].first() {
                Some(f) => f,
                None => continue,
            };

            let placement = match self.config.placements.get(slot) {
                Some(p) => p.clone(),
                None => continue,
            };

            let src_frame: &VideoFrame = match frame {
                FilterFrame::Video(v) => v,
                _ => continue,
            };

            let src_w = src_frame.width as usize;
            let src_h = src_frame.height as usize;
            let dst_w = placement.width.unwrap_or(src_frame.width) as usize;
            let dst_h = placement.height.unwrap_or(src_frame.height) as usize;
            let dst_x = placement.x as usize;
            let dst_y = placement.y as usize;
            let alpha = placement.alpha;

            // Nearest-neighbour blit of Y plane.
            for dy in 0..dst_h {
                let oy = dst_y + dy;
                if oy >= out_h {
                    break;
                }
                let sy = (dy * src_h) / dst_h.max(1);
                for dx in 0..dst_w {
                    let ox = dst_x + dx;
                    if ox >= out_w {
                        break;
                    }
                    let sx = (dx * src_w) / dst_w.max(1);

                    // Try to read from Y plane of the source frame.
                    let src_val = src_frame
                        .planes
                        .first()
                        .and_then(|p| p.data.get(sy * src_w + sx))
                        .copied()
                        .unwrap_or(16);

                    let dst_idx = oy * out_w + ox;
                    if (alpha - 1.0_f32).abs() < f32::EPSILON {
                        y_plane[dst_idx] = src_val;
                    } else {
                        let bg = y_plane[dst_idx] as f32;
                        let blended = bg + alpha * (src_val as f32 - bg);
                        y_plane[dst_idx] = blended.clamp(0.0, 255.0) as u8;
                    }
                }
            }

            // Blit UV planes (half resolution, no alpha blending).
            let uv_dst_x = dst_x / 2;
            let uv_dst_y = dst_y / 2;
            let uv_dst_w = dst_w / 2;
            let uv_dst_h = dst_h / 2;
            let uv_src_w = src_w / 2;
            let uv_src_h = src_h / 2;
            let uv_out_w = out_w / 2;
            let uv_out_h = out_h / 2;

            for dy in 0..uv_dst_h {
                let oy = uv_dst_y + dy;
                if oy >= uv_out_h {
                    break;
                }
                let sy = (dy * uv_src_h) / uv_dst_h.max(1);
                for dx in 0..uv_dst_w {
                    let ox = uv_dst_x + dx;
                    if ox >= uv_out_w {
                        break;
                    }
                    let sx = (dx * uv_src_w) / uv_dst_w.max(1);

                    let u_val = src_frame
                        .planes
                        .get(1)
                        .and_then(|p| p.data.get(sy * uv_src_w + sx))
                        .copied()
                        .unwrap_or(128);
                    let v_val = src_frame
                        .planes
                        .get(2)
                        .and_then(|p| p.data.get(sy * uv_src_w + sx))
                        .copied()
                        .unwrap_or(128);

                    u_plane[oy * uv_out_w + ox] = u_val;
                    v_plane[oy * uv_out_w + ox] = v_val;
                }
            }
        }

        // Consume one frame from each queue.
        for queue in &mut self.input_queues {
            if !queue.is_empty() {
                queue.remove(0);
            }
        }

        // Build output VideoFrame.
        use oximedia_codec::{FrameType, Plane};
        use oximedia_core::Rational;
        let out_frame = VideoFrame {
            format: PixelFormat::Yuv420p,
            width: self.config.output_width,
            height: self.config.output_height,
            planes: vec![
                Plane::with_dimensions(
                    y_plane,
                    out_w,
                    self.config.output_width,
                    self.config.output_height,
                ),
                Plane::with_dimensions(
                    u_plane,
                    out_w / 2,
                    self.config.output_width / 2,
                    self.config.output_height / 2,
                ),
                Plane::with_dimensions(
                    v_plane,
                    out_w / 2,
                    self.config.output_width / 2,
                    self.config.output_height / 2,
                ),
            ],
            timestamp: oximedia_core::Timestamp::new(0, Rational::new(1, 1000)),
            frame_type: FrameType::Key,
            color_info: oximedia_codec::ColorInfo::default(),
            corrupt: false,
        };

        Ok(FilterFrame::Video(out_frame))
    }
}

impl Node for MergeFilter {
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

    /// Process by pushing `input` into slot 0 and compositing if ready.
    ///
    /// For multi-input composition, push frames to slots 1..N via
    /// [`MergeFilter::push_input`] before calling `process`.
    fn process(&mut self, input: Option<FilterFrame>) -> GraphResult<Option<FilterFrame>> {
        if let Some(frame) = input {
            self.push_input(0, frame)?;
        }
        if self.is_ready() {
            Ok(Some(self.composite()?))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_video_frame(w: u32, h: u32) -> FilterFrame {
        FilterFrame::Video(VideoFrame::new(PixelFormat::Yuv420p, w, h))
    }

    #[test]
    fn test_merge_creation_two_inputs() {
        let config = MergeConfig::tiled(2, 1920, 1080);
        let merge = MergeFilter::new(NodeId(0), "pip", config);
        assert_eq!(merge.input_count(), 2);
        assert_eq!(merge.inputs().len(), 2);
        assert_eq!(merge.outputs().len(), 1);
    }

    #[test]
    fn test_merge_pip_config() {
        let config = MergeConfig::picture_in_picture(1920, 1080, 100, 100, 480, 270);
        assert_eq!(config.input_count(), 2);
        assert_eq!(config.output_width, 1920);
        assert_eq!(config.output_height, 1080);
    }

    #[test]
    fn test_merge_not_ready_with_empty_queues() {
        let config = MergeConfig::tiled(2, 640, 480);
        let merge = MergeFilter::new(NodeId(0), "m", config);
        assert!(!merge.is_ready());
    }

    #[test]
    fn test_merge_ready_after_all_inputs_pushed() {
        let config = MergeConfig::tiled(2, 640, 480);
        let mut merge = MergeFilter::new(NodeId(0), "m", config);
        merge
            .push_input(0, make_video_frame(320, 480))
            .expect("push should succeed");
        assert!(!merge.is_ready());
        merge
            .push_input(1, make_video_frame(320, 480))
            .expect("push should succeed");
        assert!(merge.is_ready());
    }

    #[test]
    fn test_merge_process_produces_output() {
        let config = MergeConfig::tiled(2, 640, 480);
        let mut merge = MergeFilter::new(NodeId(0), "m", config);
        merge
            .push_input(0, make_video_frame(320, 480))
            .expect("push should succeed");
        merge
            .push_input(1, make_video_frame(320, 480))
            .expect("push should succeed");
        let result = merge.process(None).expect("process should succeed");
        assert!(result.is_some());
        if let Some(FilterFrame::Video(v)) = result {
            assert_eq!(v.width, 640);
            assert_eq!(v.height, 480);
        } else {
            panic!("expected video frame");
        }
    }

    #[test]
    fn test_merge_process_without_all_inputs_returns_none() {
        let config = MergeConfig::tiled(2, 640, 480);
        let mut merge = MergeFilter::new(NodeId(0), "m", config);
        merge
            .push_input(0, make_video_frame(320, 480))
            .expect("push should succeed");
        let result = merge.process(None).expect("process should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_merge_push_invalid_port_returns_error() {
        let config = MergeConfig::tiled(2, 640, 480);
        let mut merge = MergeFilter::new(NodeId(0), "m", config);
        let result = merge.push_input(99, make_video_frame(320, 480));
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_input_port_names() {
        let config = MergeConfig::tiled(3, 1920, 1080);
        let merge = MergeFilter::new(NodeId(0), "m", config);
        assert_eq!(merge.inputs()[0].name, "input_0");
        assert_eq!(merge.inputs()[1].name, "input_1");
        assert_eq!(merge.inputs()[2].name, "input_2");
    }

    #[test]
    fn test_merge_node_type_is_filter() {
        let config = MergeConfig::tiled(2, 640, 480);
        let merge = MergeFilter::new(NodeId(0), "m", config);
        assert_eq!(merge.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_merge_placement_at() {
        let p = InputPlacement::at(10, 20);
        assert_eq!(p.x, 10);
        assert_eq!(p.y, 20);
        assert!((p.alpha - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_merge_placement_with_size() {
        let p = InputPlacement::at(0, 0).with_size(480, 270);
        assert_eq!(p.width, Some(480));
        assert_eq!(p.height, Some(270));
    }

    #[test]
    fn test_merge_placement_alpha_clamp() {
        let p = InputPlacement::default().with_alpha(1.5);
        assert!((p.alpha - 1.0).abs() < 1e-6);
        let p2 = InputPlacement::default().with_alpha(-0.5);
        assert!((p2.alpha).abs() < 1e-6);
    }

    #[test]
    fn test_merge_state_transitions() {
        let config = MergeConfig::tiled(2, 640, 480);
        let mut merge = MergeFilter::new(NodeId(0), "m", config);
        merge
            .set_state(NodeState::Processing)
            .expect("state transition should succeed");
        assert_eq!(merge.state(), NodeState::Processing);
    }
}
