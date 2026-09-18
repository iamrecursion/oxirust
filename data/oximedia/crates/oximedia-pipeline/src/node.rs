//! Pipeline node types: identifiers, stream specifications, and node configurations.

use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

// ── NodeId ────────────────────────────────────────────────────────────────────

/// Opaque identifier for a node in a `PipelineGraph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeId(Uuid);

impl NodeId {
    /// Create a new random `NodeId`.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Return the underlying `Uuid`.
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── PadId ─────────────────────────────────────────────────────────────────────

/// Identifies a named pad (port) on a specific node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PadId {
    /// The node this pad belongs to.
    pub node_id: NodeId,
    /// The pad's name (e.g. `"default"`, `"video"`, `"audio"`).
    pub pad_name: String,
}

impl PadId {
    /// Construct a `PadId`.
    pub fn new(node_id: NodeId, pad_name: impl Into<String>) -> Self {
        Self {
            node_id,
            pad_name: pad_name.into(),
        }
    }
}

impl fmt::Display for PadId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.node_id, self.pad_name)
    }
}

// ── StreamKind ────────────────────────────────────────────────────────────────

/// High-level media stream category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StreamKind {
    /// Pixel-based video stream.
    Video,
    /// PCM or compressed audio stream.
    Audio,
    /// Generic binary data stream.
    Data,
    /// Subtitle / caption stream.
    Subtitle,
}

impl fmt::Display for StreamKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            StreamKind::Video => "video",
            StreamKind::Audio => "audio",
            StreamKind::Data => "data",
            StreamKind::Subtitle => "subtitle",
        };
        write!(f, "{s}")
    }
}

// ── FrameFormat ───────────────────────────────────────────────────────────────

/// Pixel / sample format for frames passing through a pipeline pad.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FrameFormat {
    // ── Video pixel formats ──────────────────────────────────────────────────
    /// Planar YUV 4:2:0, 8-bit (most common for H.264/AV1).
    Yuv420p,
    /// Planar YUV 4:2:2, 8-bit.
    Yuv422p,
    /// Planar YUV 4:4:4, 8-bit (lossless-friendly).
    Yuv444p,
    /// Packed RGB 24-bit (R, G, B bytes).
    Rgb24,
    /// Packed RGBA 32-bit (R, G, B, A bytes).
    Rgba32,
    /// Semi-planar YUV 4:2:0, NV12 (Y plane + interleaved UV).
    Nv12,
    // ── Audio sample formats ─────────────────────────────────────────────────
    /// 32-bit float, planar (separate channel planes).
    Float32Planar,
    /// 16-bit signed integer, interleaved (all channels per sample).
    S16Interleaved,
    /// 32-bit float, interleaved.
    F32Interleaved,
}

impl FrameFormat {
    /// Returns `true` if this format is a video pixel format.
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            FrameFormat::Yuv420p
                | FrameFormat::Yuv422p
                | FrameFormat::Yuv444p
                | FrameFormat::Rgb24
                | FrameFormat::Rgba32
                | FrameFormat::Nv12
        )
    }

    /// Returns `true` if this format is an audio sample format.
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            FrameFormat::Float32Planar | FrameFormat::S16Interleaved | FrameFormat::F32Interleaved
        )
    }

    /// Bytes per sample or pixel (approximate; planar formats use per-plane accounting).
    pub fn bytes_per_element(&self) -> u32 {
        match self {
            FrameFormat::Yuv420p
            | FrameFormat::Yuv422p
            | FrameFormat::Yuv444p
            | FrameFormat::Nv12 => 1,
            FrameFormat::Rgb24 => 3,
            FrameFormat::Rgba32 => 4,
            FrameFormat::Float32Planar | FrameFormat::F32Interleaved => 4,
            FrameFormat::S16Interleaved => 2,
        }
    }
}

impl fmt::Display for FrameFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FrameFormat::Yuv420p => "yuv420p",
            FrameFormat::Yuv422p => "yuv422p",
            FrameFormat::Yuv444p => "yuv444p",
            FrameFormat::Rgb24 => "rgb24",
            FrameFormat::Rgba32 => "rgba32",
            FrameFormat::Nv12 => "nv12",
            FrameFormat::Float32Planar => "fltp",
            FrameFormat::S16Interleaved => "s16",
            FrameFormat::F32Interleaved => "flt",
        };
        write!(f, "{s}")
    }
}

// ── StreamSpec ────────────────────────────────────────────────────────────────

/// Full specification of a media stream flowing along a pipeline edge.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamSpec {
    /// Whether this is a video, audio, data, or subtitle stream.
    pub kind: StreamKind,
    /// Pixel / sample format.
    pub format: FrameFormat,
    /// Frame width in pixels (video only).
    pub width: Option<u32>,
    /// Frame height in pixels (video only).
    pub height: Option<u32>,
    /// Audio sample rate in Hz (audio only).
    pub sample_rate: Option<u32>,
    /// Number of audio channels (audio only).
    pub channels: Option<u8>,
    /// Rational time base as (numerator, denominator).  E.g. `(1, 90000)`.
    pub time_base: (u32, u32),
}

impl StreamSpec {
    /// Build a basic video `StreamSpec`.
    pub fn video(format: FrameFormat, width: u32, height: u32, fps_num: u32) -> Self {
        Self {
            kind: StreamKind::Video,
            format,
            width: Some(width),
            height: Some(height),
            sample_rate: None,
            channels: None,
            time_base: (1, fps_num),
        }
    }

    /// Build a basic audio `StreamSpec`.
    pub fn audio(format: FrameFormat, sample_rate: u32, channels: u8) -> Self {
        Self {
            kind: StreamKind::Audio,
            format,
            width: None,
            height: None,
            sample_rate: Some(sample_rate),
            channels: Some(channels),
            time_base: (1, sample_rate),
        }
    }

    /// Returns `true` when both specs carry the same `StreamKind`.
    pub fn kind_compatible(&self, other: &StreamSpec) -> bool {
        self.kind == other.kind
    }
}

impl Default for StreamSpec {
    fn default() -> Self {
        Self {
            kind: StreamKind::Video,
            format: FrameFormat::Yuv420p,
            width: None,
            height: None,
            sample_rate: None,
            channels: None,
            time_base: (1, 90000),
        }
    }
}

// ── SyntheticSource ───────────────────────────────────────────────────────────

/// Describes a synthetically-generated media source (no file or network).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SyntheticSource {
    /// Generate silent audio.
    Silence {
        /// Number of output channels.
        channels: u8,
        /// Sample rate in Hz.
        sample_rate: u32,
    },
    /// Generate a solid-black video frame.
    BlackFrame {
        /// Frame width in pixels.
        width: u32,
        /// Frame height in pixels.
        height: u32,
        /// Frames per second.
        fps: f32,
    },
    /// Generate a test pattern (SMPTE colour bars, etc.).
    TestPattern {
        /// Pattern variant index (0 = colour bars, 1 = luma ramp, …).
        pattern: u8,
    },
}

// ── SourceConfig ──────────────────────────────────────────────────────────────

/// Configuration for a source node.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SourceConfig {
    /// Read from a filesystem path.
    File(String),
    /// Read from a network URL (RTMP, HLS, …).
    Network(String),
    /// Synthesize media without an external source.
    Synthetic(SyntheticSource),
}

// ── SinkConfig ────────────────────────────────────────────────────────────────

/// Configuration for a sink node.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SinkConfig {
    /// Write encoded output to a file.
    File(String),
    /// Discard all frames (useful for benchmarking).
    Null,
    /// Write frames to an in-memory buffer identified by a label.
    Memory(String),
}

// ── FilterConfig ──────────────────────────────────────────────────────────────

/// All built-in filter variants that a `Filter` node may execute.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FilterConfig {
    /// Resize video to an exact resolution.
    Scale { width: u32, height: u32 },
    /// Crop a rectangular region from a video frame.
    Crop { x: u32, y: u32, w: u32, h: u32 },
    /// Trim the stream to a time window.
    Trim { start_ms: i64, end_ms: i64 },
    /// Adjust audio gain.
    Volume {
        /// Gain in decibels (0.0 = unity, positive = louder, negative = quieter).
        gain_db: f32,
    },
    /// Force a constant output frame-rate.
    Fps { fps: f32 },
    /// Convert pixel / sample format.
    Format(FrameFormat),
    /// Composite a secondary stream on top of a primary stream.
    Overlay,
    /// Concatenate multiple streams end-to-end.
    Concat {
        /// Number of input segments to concatenate.
        count: u32,
    },
    /// Pad the video canvas to a target resolution with black fill.
    Pad { width: u32, height: u32 },
    /// Flip video horizontally (mirror).
    Hflip,
    /// Flip video vertically (upside-down).
    Vflip,
    /// Rotate / transpose video by a multiple of 90°.
    Transpose(u8),
    /// A user-defined filter identified by name with arbitrary key=value params.
    Custom {
        /// Filter name (e.g. `"eq"`, `"unsharp"`).
        name: String,
        /// Ordered list of `(key, value)` parameter pairs.
        params: Vec<(String, String)>,
    },
    /// A parametric filter with a base configuration plus additional key-value properties.
    ///
    /// This extends any base filter with runtime-configurable properties, enabling
    /// external tooling to attach metadata, tuning knobs, or vendor-specific parameters
    /// without modifying the core `FilterConfig` enum.
    Parametric {
        /// The base filter configuration that this parametric wrapper extends.
        base: Box<FilterConfig>,
        /// Additional key-value properties that augment the base filter.
        ///
        /// Common keys: `"quality"`, `"preset"`, `"threads"`, `"vendor_ext"`.
        properties: HashMap<String, String>,
    },
}

impl FilterConfig {
    /// Create a `Parametric` filter wrapping the given base config with additional properties.
    pub fn parametric(base: FilterConfig, properties: HashMap<String, String>) -> Self {
        FilterConfig::Parametric {
            base: Box::new(base),
            properties,
        }
    }

    /// If this is a `Parametric` config, return a reference to its property map.
    pub fn properties(&self) -> Option<&HashMap<String, String>> {
        match self {
            FilterConfig::Parametric { properties, .. } => Some(properties),
            _ => None,
        }
    }

    /// If this is a `Parametric` config, return a mutable reference to its property map.
    pub fn properties_mut(&mut self) -> Option<&mut HashMap<String, String>> {
        match self {
            FilterConfig::Parametric { properties, .. } => Some(properties),
            _ => None,
        }
    }

    /// If this is a `Parametric` config, return a reference to its base config.
    pub fn base_config(&self) -> Option<&FilterConfig> {
        match self {
            FilterConfig::Parametric { base, .. } => Some(base),
            _ => None,
        }
    }

    /// Look up a single property by key. Returns `None` for non-`Parametric`
    /// configs or when the key is absent.
    pub fn get_property(&self, key: &str) -> Option<&str> {
        match self {
            FilterConfig::Parametric { properties, .. } => properties.get(key).map(|s| s.as_str()),
            _ => None,
        }
    }

    /// Builder-style method: wrap this filter in a `Parametric` envelope
    /// (or add to an existing one) and insert a single key-value property.
    ///
    /// Calling this on a non-`Parametric` config promotes it to `Parametric`
    /// with the current config as the base. Calling it on an already-`Parametric`
    /// config simply inserts the key.
    pub fn with_property(self, key: impl Into<String>, value: impl Into<String>) -> Self {
        match self {
            FilterConfig::Parametric {
                base,
                mut properties,
            } => {
                properties.insert(key.into(), value.into());
                FilterConfig::Parametric { base, properties }
            }
            other => {
                let mut properties = HashMap::new();
                properties.insert(key.into(), value.into());
                FilterConfig::Parametric {
                    base: Box::new(other),
                    properties,
                }
            }
        }
    }

    /// Returns `true` when this filter is a no-operation (identity transform).
    ///
    /// Used by `PipelineOptimizer::eliminate_noop`.
    pub fn is_noop(&self) -> bool {
        match self {
            FilterConfig::Volume { gain_db } => (*gain_db - 0.0_f32).abs() < f32::EPSILON,
            FilterConfig::Fps { fps } => *fps <= 0.0,
            FilterConfig::Trim { start_ms, end_ms } => start_ms == end_ms,
            FilterConfig::Parametric { base, .. } => base.is_noop(),
            _ => false,
        }
    }

    /// Relative computational cost estimate (lower = cheaper).
    ///
    /// Used by `PipelineOptimizer::reorder_filters`.
    pub fn cost_estimate(&self) -> u32 {
        match self {
            FilterConfig::Hflip | FilterConfig::Vflip => 1,
            FilterConfig::Transpose(_) => 2,
            FilterConfig::Volume { .. } => 2,
            FilterConfig::Format(_) => 3,
            FilterConfig::Fps { .. } => 4,
            FilterConfig::Trim { .. } => 4,
            FilterConfig::Crop { .. } => 5,
            FilterConfig::Pad { .. } => 6,
            FilterConfig::Scale { .. } => 8,
            FilterConfig::Concat { .. } => 10,
            FilterConfig::Overlay => 12,
            FilterConfig::Custom { .. } => 15,
            FilterConfig::Parametric { base, .. } => base.cost_estimate(),
        }
    }
}

// ── NodeType ──────────────────────────────────────────────────────────────────

/// Discriminant for what role a node plays in the pipeline.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NodeType {
    /// Produces frames; has no input pads.
    Source(SourceConfig),
    /// Consumes frames; has no output pads.
    Sink(SinkConfig),
    /// Transforms a stream from inputs to outputs.
    Filter(FilterConfig),
    /// Copies one input stream to N identical output streams.
    Split,
    /// Combines N input streams into one (e.g. mux audio + video).
    Merge,
    /// Passes data through unchanged (useful as a placeholder).
    Null,
    /// A conditional branching node that routes frames to `"then"` or `"else"`
    /// based on a stream property predicate.
    Conditional(IfNode),
}

// ── NodeSpec ──────────────────────────────────────────────────────────────────

/// Complete specification for a single node in a `PipelineGraph`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeSpec {
    /// Unique identifier assigned when the node is added to a graph.
    pub id: NodeId,
    /// Human-readable label for debugging and visualisation.
    pub name: String,
    /// What the node does.
    pub node_type: NodeType,
    /// Named input pads, each with their expected `StreamSpec`.
    pub input_pads: Vec<(String, StreamSpec)>,
    /// Named output pads, each with their produced `StreamSpec`.
    pub output_pads: Vec<(String, StreamSpec)>,
}

impl NodeSpec {
    /// Create a `NodeSpec` with a given type and auto-generated id.
    pub fn new(
        name: impl Into<String>,
        node_type: NodeType,
        input_pads: Vec<(String, StreamSpec)>,
        output_pads: Vec<(String, StreamSpec)>,
    ) -> Self {
        Self {
            id: NodeId::new(),
            name: name.into(),
            node_type,
            input_pads,
            output_pads,
        }
    }

    /// Convenience: create a source node with a single unnamed output pad.
    pub fn source(name: impl Into<String>, config: SourceConfig, out_spec: StreamSpec) -> Self {
        let id = NodeId::new();
        Self {
            id,
            name: name.into(),
            node_type: NodeType::Source(config),
            input_pads: vec![],
            output_pads: vec![("default".to_string(), out_spec)],
        }
    }

    /// Convenience: create a sink node with a single unnamed input pad.
    pub fn sink(name: impl Into<String>, config: SinkConfig, in_spec: StreamSpec) -> Self {
        let id = NodeId::new();
        Self {
            id,
            name: name.into(),
            node_type: NodeType::Sink(config),
            input_pads: vec![("default".to_string(), in_spec)],
            output_pads: vec![],
        }
    }

    /// Convenience: create a filter node with a single in/out pad pair.
    pub fn filter(
        name: impl Into<String>,
        config: FilterConfig,
        in_spec: StreamSpec,
        out_spec: StreamSpec,
    ) -> Self {
        let id = NodeId::new();
        Self {
            id,
            name: name.into(),
            node_type: NodeType::Filter(config),
            input_pads: vec![("default".to_string(), in_spec)],
            output_pads: vec![("default".to_string(), out_spec)],
        }
    }

    /// Convenience: create a conditional (`IfNode`) branching node.
    ///
    /// The node has a single `"default"` input pad and two output pads:
    /// - `"then"` — emitted when the condition holds.
    /// - `"else"` — emitted when the condition does not hold.
    ///
    /// Both output pads carry the same stream spec as the input.
    pub fn conditional(
        name: impl Into<String>,
        condition: IfNode,
        stream_spec: StreamSpec,
    ) -> Self {
        let id = NodeId::new();
        Self {
            id,
            name: name.into(),
            node_type: NodeType::Conditional(condition),
            input_pads: vec![("default".to_string(), stream_spec.clone())],
            output_pads: vec![
                ("then".to_string(), stream_spec.clone()),
                ("else".to_string(), stream_spec),
            ],
        }
    }
}

// ── ConditionOp ───────────────────────────────────────────────────────────────

/// A comparison operator used in [`IfNode`] stream-property conditions.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ConditionOp {
    /// Property equals the given value (string comparison).
    Eq(String),
    /// Property does not equal the given value.
    Ne(String),
    /// Numeric property is greater than the given value.
    Gt(f64),
    /// Numeric property is less than the given value.
    Lt(f64),
    /// Numeric property is greater than or equal to the given value.
    Ge(f64),
    /// Numeric property is less than or equal to the given value.
    Le(f64),
    /// The property exists (has any value).
    Exists,
    /// The property does not exist.
    NotExists,
}

impl ConditionOp {
    /// Evaluate the condition against a string value extracted from the stream.
    ///
    /// For numeric operators (`Gt`, `Lt`, `Ge`, `Le`), the value is parsed
    /// as `f64`; if parsing fails the condition evaluates to `false`.
    pub fn evaluate(&self, value: Option<&str>) -> bool {
        match self {
            ConditionOp::Exists => value.is_some(),
            ConditionOp::NotExists => value.is_none(),
            ConditionOp::Eq(expected) => value.map(|v| v == expected.as_str()).unwrap_or(false),
            ConditionOp::Ne(expected) => value.map(|v| v != expected.as_str()).unwrap_or(false),
            ConditionOp::Gt(threshold) => value
                .and_then(|v| v.parse::<f64>().ok())
                .map(|n| n > *threshold)
                .unwrap_or(false),
            ConditionOp::Lt(threshold) => value
                .and_then(|v| v.parse::<f64>().ok())
                .map(|n| n < *threshold)
                .unwrap_or(false),
            ConditionOp::Ge(threshold) => value
                .and_then(|v| v.parse::<f64>().ok())
                .map(|n| n >= *threshold)
                .unwrap_or(false),
            ConditionOp::Le(threshold) => value
                .and_then(|v| v.parse::<f64>().ok())
                .map(|n| n <= *threshold)
                .unwrap_or(false),
        }
    }
}

impl fmt::Display for ConditionOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConditionOp::Eq(v) => write!(f, "== {v}"),
            ConditionOp::Ne(v) => write!(f, "!= {v}"),
            ConditionOp::Gt(v) => write!(f, "> {v}"),
            ConditionOp::Lt(v) => write!(f, "< {v}"),
            ConditionOp::Ge(v) => write!(f, ">= {v}"),
            ConditionOp::Le(v) => write!(f, "<= {v}"),
            ConditionOp::Exists => write!(f, "exists"),
            ConditionOp::NotExists => write!(f, "not_exists"),
        }
    }
}

// ── StreamProperty ────────────────────────────────────────────────────────────

/// A stream property key that `IfNode` can inspect.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StreamProperty {
    /// The stream kind: `"video"`, `"audio"`, `"data"`, or `"subtitle"`.
    Kind,
    /// Video frame width in pixels (serialised as decimal string).
    Width,
    /// Video frame height in pixels (serialised as decimal string).
    Height,
    /// Audio sample rate in Hz (serialised as decimal string).
    SampleRate,
    /// Number of audio channels (serialised as decimal string).
    Channels,
    /// The pixel / sample format (e.g. `"yuv420p"`, `"s16"`).
    Format,
    /// A custom user-defined property key.
    Custom(String),
}

impl StreamProperty {
    /// Extract the value of this property from a [`StreamSpec`] as a string.
    ///
    /// Returns `None` when the property is not applicable to the given spec
    /// (e.g. requesting `Width` on an audio stream).
    pub fn extract<'a>(&'a self, spec: &'a StreamSpec) -> Option<String> {
        match self {
            StreamProperty::Kind => Some(spec.kind.to_string()),
            StreamProperty::Format => Some(spec.format.to_string()),
            StreamProperty::Width => spec.width.map(|w| w.to_string()),
            StreamProperty::Height => spec.height.map(|h| h.to_string()),
            StreamProperty::SampleRate => spec.sample_rate.map(|r| r.to_string()),
            StreamProperty::Channels => spec.channels.map(|c| c.to_string()),
            StreamProperty::Custom(_) => None,
        }
    }
}

impl fmt::Display for StreamProperty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamProperty::Kind => write!(f, "kind"),
            StreamProperty::Width => write!(f, "width"),
            StreamProperty::Height => write!(f, "height"),
            StreamProperty::SampleRate => write!(f, "sample_rate"),
            StreamProperty::Channels => write!(f, "channels"),
            StreamProperty::Format => write!(f, "format"),
            StreamProperty::Custom(k) => write!(f, "{k}"),
        }
    }
}

// ── IfNode ────────────────────────────────────────────────────────────────────

/// A conditional branching node that routes frames to one of two outputs based
/// on a stream property predicate.
///
/// When the predicate evaluates to `true` the frame is emitted on the `"then"`
/// output pad; otherwise it is emitted on the `"else"` output pad.  This
/// allows declarative branching in a pipeline without hard-coding runtime logic
/// in the filter graph.
///
/// # Integration with `NodeSpec`
///
/// An `IfNode` is stored inside a [`NodeType::Conditional`] variant.  The
/// builder inserts it with two output pads named `"then"` and `"else"`.
///
/// # Example
///
/// ```rust
/// use oximedia_pipeline::node::{IfNode, ConditionOp, StreamProperty};
/// use oximedia_pipeline::node::{StreamSpec, FrameFormat};
///
/// let condition = IfNode::new(StreamProperty::Width, ConditionOp::Ge(1280.0));
/// let spec = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
/// assert!(condition.evaluate(&spec));
///
/// let spec_low = StreamSpec::video(FrameFormat::Yuv420p, 640, 480, 25);
/// assert!(!condition.evaluate(&spec_low));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IfNode {
    /// The stream property to inspect.
    pub property: StreamProperty,
    /// The comparison operator applied to the property value.
    pub op: ConditionOp,
    /// Human-readable description of the condition (auto-generated if empty).
    pub description: String,
}

impl IfNode {
    /// Create a new `IfNode` with the given property and operator.
    pub fn new(property: StreamProperty, op: ConditionOp) -> Self {
        let desc = format!("{property} {op}");
        Self {
            property,
            op,
            description: desc,
        }
    }

    /// Create a new `IfNode` with an explicit human-readable description.
    pub fn with_description(
        property: StreamProperty,
        op: ConditionOp,
        description: impl Into<String>,
    ) -> Self {
        Self {
            property,
            op,
            description: description.into(),
        }
    }

    /// Evaluate the condition against the given [`StreamSpec`].
    ///
    /// Returns `true` if the predicate holds (route to `"then"` pad).
    pub fn evaluate(&self, spec: &StreamSpec) -> bool {
        let value = self.property.extract(spec);
        self.op.evaluate(value.as_deref())
    }
}

impl fmt::Display for IfNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.description.is_empty() {
            write!(f, "if {} {}", self.property, self.op)
        } else {
            write!(f, "if({})", self.description)
        }
    }
}

// ── NodeType (extended) ───────────────────────────────────────────────────────
// NOTE: NodeType is defined earlier in the file; we extend it here via the
// Conditional variant by modifying the enum declaration above.

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_uniqueness() {
        let a = NodeId::new();
        let b = NodeId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn node_id_display() {
        let id = NodeId::new();
        let s = id.to_string();
        // UUID format: 8-4-4-4-12 hex chars + 4 hyphens = 36 chars
        assert_eq!(s.len(), 36);
    }

    #[test]
    fn pad_id_display() {
        let node = NodeId::new();
        let pad = PadId::new(node, "video");
        let s = pad.to_string();
        assert!(s.ends_with(":video"));
    }

    #[test]
    fn stream_kind_display() {
        assert_eq!(StreamKind::Video.to_string(), "video");
        assert_eq!(StreamKind::Audio.to_string(), "audio");
        assert_eq!(StreamKind::Data.to_string(), "data");
        assert_eq!(StreamKind::Subtitle.to_string(), "subtitle");
    }

    #[test]
    fn frame_format_is_video() {
        assert!(FrameFormat::Yuv420p.is_video());
        assert!(FrameFormat::Rgb24.is_video());
        assert!(!FrameFormat::S16Interleaved.is_video());
    }

    #[test]
    fn frame_format_is_audio() {
        assert!(FrameFormat::Float32Planar.is_audio());
        assert!(FrameFormat::S16Interleaved.is_audio());
        assert!(!FrameFormat::Yuv420p.is_audio());
    }

    #[test]
    fn frame_format_display() {
        assert_eq!(FrameFormat::Yuv420p.to_string(), "yuv420p");
        assert_eq!(FrameFormat::Nv12.to_string(), "nv12");
        assert_eq!(FrameFormat::Float32Planar.to_string(), "fltp");
    }

    #[test]
    fn stream_spec_video_builder() {
        let spec = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
        assert_eq!(spec.kind, StreamKind::Video);
        assert_eq!(spec.width, Some(1920));
        assert_eq!(spec.height, Some(1080));
        assert!(spec.sample_rate.is_none());
    }

    #[test]
    fn stream_spec_audio_builder() {
        let spec = StreamSpec::audio(FrameFormat::S16Interleaved, 48000, 2);
        assert_eq!(spec.kind, StreamKind::Audio);
        assert_eq!(spec.sample_rate, Some(48000));
        assert_eq!(spec.channels, Some(2));
        assert!(spec.width.is_none());
    }

    #[test]
    fn stream_spec_kind_compatible() {
        let v = StreamSpec::video(FrameFormat::Yuv420p, 1280, 720, 30);
        let v2 = StreamSpec::video(FrameFormat::Rgb24, 640, 480, 30);
        let a = StreamSpec::audio(FrameFormat::S16Interleaved, 44100, 2);
        assert!(v.kind_compatible(&v2));
        assert!(!v.kind_compatible(&a));
    }

    #[test]
    fn node_spec_source_convenience() {
        let spec = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
        let node = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), spec);
        assert!(node.input_pads.is_empty());
        assert_eq!(node.output_pads.len(), 1);
        assert_eq!(node.output_pads[0].0, "default");
    }

    #[test]
    fn node_spec_sink_convenience() {
        let spec = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
        let node = NodeSpec::sink("sink", SinkConfig::Null, spec);
        assert!(node.output_pads.is_empty());
        assert_eq!(node.input_pads.len(), 1);
    }

    #[test]
    fn filter_config_scale_cost() {
        assert_eq!(
            FilterConfig::Scale {
                width: 1280,
                height: 720
            }
            .cost_estimate(),
            8
        );
    }

    #[test]
    fn filter_config_volume_noop() {
        assert!(FilterConfig::Volume { gain_db: 0.0 }.is_noop());
        assert!(!FilterConfig::Volume { gain_db: 3.0 }.is_noop());
    }

    #[test]
    fn filter_config_trim_noop() {
        assert!(FilterConfig::Trim {
            start_ms: 1000,
            end_ms: 1000
        }
        .is_noop());
        assert!(!FilterConfig::Trim {
            start_ms: 0,
            end_ms: 5000
        }
        .is_noop());
    }

    #[test]
    fn synthetic_source_variants() {
        let silence = SyntheticSource::Silence {
            channels: 2,
            sample_rate: 44100,
        };
        let black = SyntheticSource::BlackFrame {
            width: 1920,
            height: 1080,
            fps: 25.0,
        };
        let pattern = SyntheticSource::TestPattern { pattern: 0 };
        // Just ensure they are constructable and debug-printable
        let _ = format!("{silence:?}");
        let _ = format!("{black:?}");
        let _ = format!("{pattern:?}");
    }

    #[test]
    fn node_type_variants_debug() {
        let t = NodeType::Source(SourceConfig::File("x.mp4".into()));
        let _ = format!("{t:?}");
        let t2 = NodeType::Sink(SinkConfig::Memory("buf".into()));
        let _ = format!("{t2:?}");
    }

    #[test]
    fn frame_format_bytes_per_element() {
        assert_eq!(FrameFormat::Rgba32.bytes_per_element(), 4);
        assert_eq!(FrameFormat::Rgb24.bytes_per_element(), 3);
        assert_eq!(FrameFormat::S16Interleaved.bytes_per_element(), 2);
    }

    #[test]
    fn node_spec_filter_convenience() {
        let in_spec = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
        let out_spec = StreamSpec::video(FrameFormat::Yuv420p, 1280, 720, 25);
        let node = NodeSpec::filter(
            "scale",
            FilterConfig::Scale {
                width: 1280,
                height: 720,
            },
            in_spec,
            out_spec,
        );
        assert_eq!(node.input_pads.len(), 1);
        assert_eq!(node.output_pads.len(), 1);
    }

    #[test]
    fn filter_config_custom() {
        let f = FilterConfig::Custom {
            name: "eq".to_string(),
            params: vec![("brightness".to_string(), "0.5".to_string())],
        };
        assert_eq!(f.cost_estimate(), 15);
        assert!(!f.is_noop());
    }

    #[test]
    fn filter_config_hflip_cost() {
        assert_eq!(FilterConfig::Hflip.cost_estimate(), 1);
        assert_eq!(FilterConfig::Overlay.cost_estimate(), 12);
    }

    #[test]
    fn pad_id_equality() {
        let node = NodeId::new();
        let p1 = PadId::new(node, "audio");
        let p2 = PadId::new(node, "audio");
        assert_eq!(p1, p2);
    }

    #[test]
    fn node_id_as_uuid() {
        let id = NodeId::new();
        let u = id.as_uuid();
        assert_eq!(id.to_string(), u.to_string());
    }

    #[test]
    fn parametric_filter_wraps_base() {
        let base = FilterConfig::Scale {
            width: 1280,
            height: 720,
        };
        let mut props = HashMap::new();
        props.insert("quality".to_string(), "high".to_string());
        props.insert("preset".to_string(), "fast".to_string());
        let param = FilterConfig::parametric(base.clone(), props);

        assert_eq!(param.cost_estimate(), base.cost_estimate());
        assert!(!param.is_noop());

        let props_ref = param.properties().expect("should have properties");
        assert_eq!(props_ref.get("quality").map(|s| s.as_str()), Some("high"));
        assert_eq!(props_ref.get("preset").map(|s| s.as_str()), Some("fast"));

        let base_ref = param.base_config().expect("should have base");
        assert_eq!(*base_ref, base);
    }

    #[test]
    fn parametric_filter_noop_delegates_to_base() {
        let noop_base = FilterConfig::Volume { gain_db: 0.0 };
        let param = FilterConfig::parametric(noop_base, HashMap::new());
        assert!(param.is_noop());

        let active_base = FilterConfig::Volume { gain_db: 6.0 };
        let param2 = FilterConfig::parametric(active_base, HashMap::new());
        assert!(!param2.is_noop());
    }

    #[test]
    fn parametric_properties_mut() {
        let base = FilterConfig::Hflip;
        let mut param = FilterConfig::parametric(base, HashMap::new());
        if let Some(props) = param.properties_mut() {
            props.insert("threads".to_string(), "4".to_string());
        }
        let props_ref = param.properties().expect("should have properties");
        assert_eq!(props_ref.get("threads").map(|s| s.as_str()), Some("4"));
    }

    #[test]
    fn non_parametric_has_no_properties() {
        let f = FilterConfig::Hflip;
        assert!(f.properties().is_none());
        assert!(f.base_config().is_none());
    }

    #[test]
    fn get_property_on_parametric() {
        let f = FilterConfig::Scale {
            width: 1280,
            height: 720,
        }
        .with_property("quality", "high");
        assert_eq!(f.get_property("quality"), Some("high"));
        assert_eq!(f.get_property("missing"), None);
    }

    #[test]
    fn get_property_on_non_parametric() {
        let f = FilterConfig::Hflip;
        assert_eq!(f.get_property("anything"), None);
    }

    #[test]
    fn with_property_promotes_to_parametric() {
        let f = FilterConfig::Hflip.with_property("threads", "8");
        assert!(f.properties().is_some());
        let props = f.properties().expect("has properties");
        assert_eq!(props.get("threads").map(|s| s.as_str()), Some("8"));
        let base = f.base_config().expect("has base");
        assert_eq!(*base, FilterConfig::Hflip);
    }

    #[test]
    fn with_property_chains() {
        let f = FilterConfig::Scale {
            width: 640,
            height: 480,
        }
        .with_property("quality", "medium")
        .with_property("preset", "fast")
        .with_property("threads", "4");
        assert_eq!(f.get_property("quality"), Some("medium"));
        assert_eq!(f.get_property("preset"), Some("fast"));
        assert_eq!(f.get_property("threads"), Some("4"));
        // Base is still Scale
        let base = f.base_config().expect("has base");
        assert_eq!(
            *base,
            FilterConfig::Scale {
                width: 640,
                height: 480
            }
        );
    }

    #[test]
    fn with_property_overwrites_existing_key() {
        let f = FilterConfig::Hflip
            .with_property("quality", "low")
            .with_property("quality", "high");
        assert_eq!(f.get_property("quality"), Some("high"));
    }

    #[test]
    fn parametric_cost_delegates_to_base() {
        let f = FilterConfig::Overlay.with_property("mode", "blend");
        assert_eq!(f.cost_estimate(), FilterConfig::Overlay.cost_estimate());
    }

    #[test]
    fn parametric_noop_delegates_correctly() {
        // Volume 0.0 is noop
        let f = FilterConfig::Volume { gain_db: 0.0 }.with_property("limiter", "true");
        assert!(f.is_noop());

        // Volume 3.0 is not noop
        let f2 = FilterConfig::Volume { gain_db: 3.0 }.with_property("limiter", "true");
        assert!(!f2.is_noop());
    }

    #[test]
    fn stream_spec_default() {
        let d = StreamSpec::default();
        assert_eq!(d.kind, StreamKind::Video);
        assert_eq!(d.format, FrameFormat::Yuv420p);
        assert!(d.width.is_none());
    }

    #[test]
    fn node_spec_new_assigns_unique_ids() {
        let vs = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
        let n1 = NodeSpec::new(
            "a",
            NodeType::Null,
            vec![("in".to_string(), vs.clone())],
            vec![("out".to_string(), vs.clone())],
        );
        let n2 = NodeSpec::new(
            "b",
            NodeType::Null,
            vec![("in".to_string(), vs.clone())],
            vec![("out".to_string(), vs)],
        );
        assert_ne!(n1.id, n2.id);
    }
}
