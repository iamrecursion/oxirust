//! Pipeline serialization and deserialization helpers.
//!
//! This module provides [`PipelineSerializer`] and [`PipelineDeserializer`]
//! as ergonomic wrappers around `serde_json` for converting
//! [`PipelineGraph`] instances to and from JSON.
//!
//! The underlying `serde` derives on `PipelineGraph`, `NodeSpec`,
//! `FilterConfig`, `SourceConfig`, `SinkConfig`, and associated types are
//! enabled by the `serde` feature flag; this module is therefore only
//! compiled when `--features serde` is active.
//!
//! # Round-trip guarantee
//!
//! Any graph that can be built with [`PipelineBuilder`](crate::builder::PipelineBuilder)
//! or [`PipelineGraph`] can be serialized to JSON and deserialized back
//! without data loss.  The topological sort of the deserialized graph
//! produces an equivalent ordering to the original.
//!
//! # Example
//!
//! ```rust,ignore
//! use oximedia_pipeline::graph::PipelineGraph;
//! use oximedia_pipeline::node::{NodeSpec, SourceConfig, SinkConfig, StreamSpec, FrameFormat};
//! use oximedia_pipeline::serialization::{PipelineSerializer, PipelineDeserializer};
//!
//! let mut g = PipelineGraph::new();
//! let vs = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
//! let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs.clone());
//! let sink = NodeSpec::sink("sink", SinkConfig::Null, vs);
//! let s = g.add_node(src);
//! let sk = g.add_node(sink);
//! g.connect(s, "default", sk, "default").expect("connect");
//!
//! let json = PipelineSerializer::new().to_json(&g).expect("serialize");
//! let g2 = PipelineDeserializer::new().from_json(&json).expect("deserialize");
//! assert_eq!(g2.node_count(), 2);
//! assert_eq!(g2.edge_count(), 1);
//! ```

#![cfg(feature = "serde")]

use crate::graph::PipelineGraph;

// ── SerializationError ────────────────────────────────────────────────────────

/// Errors that can occur during pipeline serialization or deserialization.
#[derive(Debug, thiserror::Error)]
pub enum SerializationError {
    /// The underlying JSON library returned an error.
    #[error("json error: {0}")]
    JsonError(String),

    /// The JSON is valid but does not represent a valid `PipelineGraph`.
    #[error("invalid structure: {0}")]
    InvalidStructure(String),

    /// A required field is absent from the JSON input.
    #[error("missing field: {0}")]
    MissingField(String),
}

impl From<serde_json::Error> for SerializationError {
    fn from(e: serde_json::Error) -> Self {
        SerializationError::JsonError(e.to_string())
    }
}

// ── PipelineSerializer ────────────────────────────────────────────────────────

/// Serializes [`PipelineGraph`] instances to JSON strings.
///
/// # Example
///
/// ```rust,ignore
/// let json = PipelineSerializer::new().to_json(&graph).expect("ok");
/// ```
#[derive(Debug, Default, Clone)]
pub struct PipelineSerializer {
    /// When `true`, the output JSON is indented for human readability.
    pub pretty: bool,
}

impl PipelineSerializer {
    /// Create a new serializer that produces compact JSON by default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new serializer that produces human-readable (indented) JSON.
    pub fn pretty() -> Self {
        Self { pretty: true }
    }

    /// Serialize `pipeline` to a compact JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`SerializationError::JsonError`] if `serde_json` fails to
    /// serialize the graph (should be rare given all fields implement
    /// `Serialize`).
    pub fn to_json(&self, pipeline: &PipelineGraph) -> Result<String, SerializationError> {
        if self.pretty {
            serde_json::to_string_pretty(pipeline).map_err(SerializationError::from)
        } else {
            serde_json::to_string(pipeline).map_err(SerializationError::from)
        }
    }

    /// Serialize `pipeline` to a pretty-printed (indented) JSON string.
    ///
    /// This is a convenience alias for
    /// `PipelineSerializer::pretty().to_json(pipeline)`.
    pub fn to_json_pretty(&self, pipeline: &PipelineGraph) -> Result<String, SerializationError> {
        serde_json::to_string_pretty(pipeline).map_err(SerializationError::from)
    }
}

// ── PipelineDeserializer ──────────────────────────────────────────────────────

/// Deserializes [`PipelineGraph`] instances from JSON strings.
///
/// # Example
///
/// ```rust,ignore
/// let g = PipelineDeserializer::new().from_json(&json).expect("ok");
/// ```
#[derive(Debug, Default, Clone)]
pub struct PipelineDeserializer {
    /// When `true`, extra unknown JSON fields are silently ignored rather than
    /// causing a deserialization error.
    ///
    /// (Currently informational; `serde_json` ignores unknown fields by
    /// default when using `#[serde(deny_unknown_fields)]` is NOT set on the
    /// types, which is the case here.)
    pub lenient: bool,
}

impl PipelineDeserializer {
    /// Create a new deserializer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Deserialize a [`PipelineGraph`] from a JSON string.
    ///
    /// # Errors
    ///
    /// - [`SerializationError::JsonError`] — the input is not valid JSON.
    /// - [`SerializationError::InvalidStructure`] — the JSON is syntactically
    ///   valid but cannot be mapped to a `PipelineGraph`.
    pub fn from_json(&self, json: &str) -> Result<PipelineGraph, SerializationError> {
        if json.trim().is_empty() {
            return Err(SerializationError::InvalidStructure(
                "input JSON is empty".to_string(),
            ));
        }
        serde_json::from_str(json).map_err(|e: serde_json::Error| {
            if e.is_data() {
                SerializationError::InvalidStructure(e.to_string())
            } else {
                SerializationError::JsonError(e.to_string())
            }
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;
    use crate::node::{
        FilterConfig, FrameFormat, NodeSpec, NodeType, SinkConfig, SourceConfig, StreamSpec,
        SyntheticSource,
    };

    fn vs() -> StreamSpec {
        StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25)
    }

    fn audio_spec() -> StreamSpec {
        StreamSpec::audio(FrameFormat::S16Interleaved, 48000, 2)
    }

    fn ser() -> PipelineSerializer {
        PipelineSerializer::new()
    }

    fn de() -> PipelineDeserializer {
        PipelineDeserializer::new()
    }

    // ── Round-trip helpers ───────────────────────────────────────────────────

    fn roundtrip(g: &PipelineGraph) -> PipelineGraph {
        let json = ser().to_json(g).expect("serialize");
        de().from_json(&json).expect("deserialize")
    }

    // ── Empty graph ──────────────────────────────────────────────────────────

    #[test]
    fn empty_graph_roundtrip() {
        let g = PipelineGraph::new();
        let g2 = roundtrip(&g);
        assert_eq!(g2.node_count(), 0);
        assert_eq!(g2.edge_count(), 0);
    }

    // ── Source + sink ────────────────────────────────────────────────────────

    #[test]
    fn source_sink_roundtrip() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs());
        let s = g.add_node(src);
        let sk = g.add_node(sink);
        g.connect(s, "default", sk, "default").expect("connect");
        let g2 = roundtrip(&g);
        assert_eq!(g2.node_count(), 2);
        assert_eq!(g2.edge_count(), 1);
    }

    // ── Source + filter + sink ───────────────────────────────────────────────

    #[test]
    fn filter_chain_roundtrip() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let filt = NodeSpec::filter(
            "scale",
            FilterConfig::Scale {
                width: 1280,
                height: 720,
            },
            vs(),
            StreamSpec::video(FrameFormat::Yuv420p, 1280, 720, 25),
        );
        let sink = NodeSpec::sink(
            "sink",
            SinkConfig::Null,
            StreamSpec::video(FrameFormat::Yuv420p, 1280, 720, 25),
        );
        let s = g.add_node(src);
        let f = g.add_node(filt);
        let sk = g.add_node(sink);
        g.connect(s, "default", f, "default").expect("connect");
        g.connect(f, "default", sk, "default").expect("connect");
        let g2 = roundtrip(&g);
        assert_eq!(g2.node_count(), 3);
        assert_eq!(g2.edge_count(), 2);
        let sorted = g2.topological_sort().expect("no cycle");
        assert_eq!(sorted.len(), 3);
    }

    // ── Branched (fan-out) graph ─────────────────────────────────────────────

    #[test]
    fn branched_graph_roundtrip() {
        let mut g = PipelineGraph::new();
        let vs_spec = vs();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs_spec.clone());
        let split = crate::node::NodeSpec::new(
            "split",
            NodeType::Split,
            vec![("default".to_string(), vs_spec.clone())],
            vec![
                ("out0".to_string(), vs_spec.clone()),
                ("out1".to_string(), vs_spec.clone()),
            ],
        );
        let sink1 = NodeSpec::sink("sink1", SinkConfig::File("a.mp4".into()), vs_spec.clone());
        let sink2 = NodeSpec::sink("sink2", SinkConfig::File("b.mp4".into()), vs_spec);
        let s = g.add_node(src);
        let sp = g.add_node(split);
        let sk1 = g.add_node(sink1);
        let sk2 = g.add_node(sink2);
        g.connect(s, "default", sp, "default").expect("connect");
        g.connect(sp, "out0", sk1, "default").expect("connect");
        g.connect(sp, "out1", sk2, "default").expect("connect");
        let g2 = roundtrip(&g);
        assert_eq!(g2.node_count(), 4);
        assert_eq!(g2.edge_count(), 3);
    }

    // ── FilterConfig variants ────────────────────────────────────────────────

    #[test]
    fn filter_scale_roundtrip() {
        let cfg = FilterConfig::Scale {
            width: 1280,
            height: 720,
        };
        let node = NodeSpec::filter("f", cfg.clone(), vs(), vs());
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let s = g.add_node(src);
        let f = g.add_node(node);
        g.connect(s, "default", f, "default").expect("connect");
        let g2 = roundtrip(&g);
        let f_spec = g2.nodes.values().find(|n| n.name == "f").expect("found");
        assert!(matches!(
            f_spec.node_type,
            NodeType::Filter(FilterConfig::Scale {
                width: 1280,
                height: 720
            })
        ));
    }

    #[test]
    fn filter_hflip_vflip_roundtrip() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let hflip = NodeSpec::filter("hflip", FilterConfig::Hflip, vs(), vs());
        let vflip = NodeSpec::filter("vflip", FilterConfig::Vflip, vs(), vs());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs());
        let s = g.add_node(src);
        let hf = g.add_node(hflip);
        let vf = g.add_node(vflip);
        let sk = g.add_node(sink);
        g.connect(s, "default", hf, "default").expect("connect");
        g.connect(hf, "default", vf, "default").expect("connect");
        g.connect(vf, "default", sk, "default").expect("connect");
        let g2 = roundtrip(&g);
        assert_eq!(g2.node_count(), 4);
        assert_eq!(g2.edge_count(), 3);
    }

    #[test]
    fn filter_fps_volume_trim_roundtrip() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let fps = NodeSpec::filter("fps", FilterConfig::Fps { fps: 30.0 }, vs(), vs());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs());
        let s = g.add_node(src);
        let f = g.add_node(fps);
        let sk = g.add_node(sink);
        g.connect(s, "default", f, "default").expect("connect");
        g.connect(f, "default", sk, "default").expect("connect");
        let g2 = roundtrip(&g);
        let fps_spec = g2
            .nodes
            .values()
            .find(|n| n.name == "fps")
            .expect("fps node");
        assert!(
            matches!(fps_spec.node_type, NodeType::Filter(FilterConfig::Fps { fps }) if (fps - 30.0).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn filter_crop_roundtrip() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let crop = NodeSpec::filter(
            "crop",
            FilterConfig::Crop {
                x: 10,
                y: 20,
                w: 640,
                h: 480,
            },
            vs(),
            StreamSpec::video(FrameFormat::Yuv420p, 640, 480, 25),
        );
        let sink = NodeSpec::sink(
            "sink",
            SinkConfig::Null,
            StreamSpec::video(FrameFormat::Yuv420p, 640, 480, 25),
        );
        let s = g.add_node(src);
        let c = g.add_node(crop);
        let sk = g.add_node(sink);
        g.connect(s, "default", c, "default").expect("connect");
        g.connect(c, "default", sk, "default").expect("connect");
        let g2 = roundtrip(&g);
        let crop_spec = g2.nodes.values().find(|n| n.name == "crop").expect("crop");
        assert!(matches!(
            crop_spec.node_type,
            NodeType::Filter(FilterConfig::Crop {
                x: 10,
                y: 20,
                w: 640,
                h: 480
            })
        ));
    }

    #[test]
    fn filter_custom_roundtrip() {
        let custom = FilterConfig::Custom {
            name: "eq".to_string(),
            params: vec![("brightness".to_string(), "0.5".to_string())],
        };
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let f = NodeSpec::filter("eq", custom, vs(), vs());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs());
        let s = g.add_node(src);
        let fid = g.add_node(f);
        let sk = g.add_node(sink);
        g.connect(s, "default", fid, "default").expect("connect");
        g.connect(fid, "default", sk, "default").expect("connect");
        let g2 = roundtrip(&g);
        let eq_spec = g2.nodes.values().find(|n| n.name == "eq").expect("eq node");
        if let NodeType::Filter(FilterConfig::Custom { name, params }) = &eq_spec.node_type {
            assert_eq!(name, "eq");
            assert_eq!(params.len(), 1);
        } else {
            panic!("expected Custom filter");
        }
    }

    #[test]
    fn filter_parametric_roundtrip() {
        let base = FilterConfig::Scale {
            width: 1280,
            height: 720,
        };
        let param = base
            .with_property("quality", "high")
            .with_property("threads", "4");
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let f = NodeSpec::filter("pscale", param, vs(), vs());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs());
        let s = g.add_node(src);
        let fid = g.add_node(f);
        let sk = g.add_node(sink);
        g.connect(s, "default", fid, "default").expect("connect");
        g.connect(fid, "default", sk, "default").expect("connect");
        let g2 = roundtrip(&g);
        let ps = g2
            .nodes
            .values()
            .find(|n| n.name == "pscale")
            .expect("pscale");
        if let NodeType::Filter(ref cfg) = ps.node_type {
            assert_eq!(cfg.get_property("quality"), Some("high"));
            assert_eq!(cfg.get_property("threads"), Some("4"));
        } else {
            panic!("expected Filter node");
        }
    }

    // ── SourceConfig variants ────────────────────────────────────────────────

    #[test]
    fn source_config_network_roundtrip() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source(
            "live",
            SourceConfig::Network("rtmp://example.com/live".into()),
            vs(),
        );
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs());
        let s = g.add_node(src);
        let sk = g.add_node(sink);
        g.connect(s, "default", sk, "default").expect("connect");
        let g2 = roundtrip(&g);
        let live = g2.nodes.values().find(|n| n.name == "live").expect("live");
        assert!(matches!(
            live.node_type,
            NodeType::Source(SourceConfig::Network(_))
        ));
    }

    #[test]
    fn source_config_synthetic_roundtrip() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source(
            "test_src",
            SourceConfig::Synthetic(SyntheticSource::BlackFrame {
                width: 1920,
                height: 1080,
                fps: 25.0,
            }),
            vs(),
        );
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs());
        let s = g.add_node(src);
        let sk = g.add_node(sink);
        g.connect(s, "default", sk, "default").expect("connect");
        let g2 = roundtrip(&g);
        let tsrc = g2
            .nodes
            .values()
            .find(|n| n.name == "test_src")
            .expect("test_src");
        assert!(matches!(
            tsrc.node_type,
            NodeType::Source(SourceConfig::Synthetic(_))
        ));
    }

    // ── SinkConfig variants ──────────────────────────────────────────────────

    #[test]
    fn sink_config_memory_roundtrip() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let sink = NodeSpec::sink("mem_sink", SinkConfig::Memory("buffer0".into()), vs());
        let s = g.add_node(src);
        let sk = g.add_node(sink);
        g.connect(s, "default", sk, "default").expect("connect");
        let g2 = roundtrip(&g);
        let msink = g2
            .nodes
            .values()
            .find(|n| n.name == "mem_sink")
            .expect("mem_sink");
        assert!(matches!(
            msink.node_type,
            NodeType::Sink(SinkConfig::Memory(_))
        ));
    }

    // ── Error cases ──────────────────────────────────────────────────────────

    #[test]
    fn deserialize_invalid_json_returns_error() {
        let result = de().from_json("{ this is not valid json }");
        assert!(result.is_err());
        if let Err(SerializationError::JsonError(_)) = result {
            // expected
        } else if let Err(SerializationError::InvalidStructure(_)) = result {
            // also acceptable
        } else {
            panic!("expected Json or InvalidStructure error");
        }
    }

    #[test]
    fn deserialize_empty_string_returns_error() {
        let result = de().from_json("");
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(SerializationError::InvalidStructure(_))
        ));
    }

    #[test]
    fn deserialize_valid_json_wrong_structure_returns_error() {
        // Valid JSON object but missing required graph fields
        let result = de().from_json(r#"{"foo": "bar", "baz": 42}"#);
        // serde should fail since 'nodes' and 'edges' fields are required
        assert!(result.is_err());
    }

    // ── Pretty-print ─────────────────────────────────────────────────────────

    #[test]
    fn pretty_print_produces_valid_json() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs());
        let s = g.add_node(src);
        let sk = g.add_node(sink);
        g.connect(s, "default", sk, "default").expect("connect");

        let pretty_json = ser().to_json_pretty(&g).expect("pretty ok");
        // Must be parseable
        let g2 = de().from_json(&pretty_json).expect("deserialize pretty");
        assert_eq!(g2.node_count(), 2);
        assert_eq!(g2.edge_count(), 1);
        // Pretty JSON should contain newlines
        assert!(pretty_json.contains('\n'));
    }

    #[test]
    fn pretty_serializer_flag() {
        let g = PipelineGraph::new();
        let compact = PipelineSerializer::new().to_json(&g).expect("compact");
        let pretty = PipelineSerializer::pretty().to_json(&g).expect("pretty");
        // Both should parse to the same empty graph
        let g1 = de().from_json(&compact).expect("compact de");
        let g2 = de().from_json(&pretty).expect("pretty de");
        assert_eq!(g1.node_count(), g2.node_count());
    }

    // ── Topo sort after roundtrip ────────────────────────────────────────────

    #[test]
    fn topo_sort_works_after_roundtrip() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let filt = NodeSpec::filter(
            "scale",
            FilterConfig::Scale {
                width: 1280,
                height: 720,
            },
            vs(),
            vs(),
        );
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs());
        let s = g.add_node(src);
        let f = g.add_node(filt);
        let sk = g.add_node(sink);
        g.connect(s, "default", f, "default").expect("connect");
        g.connect(f, "default", sk, "default").expect("connect");

        let g2 = roundtrip(&g);
        let sorted = g2.topological_sort().expect("no cycle after roundtrip");
        assert_eq!(sorted.len(), 3);
    }

    // ── Audio stream roundtrip ───────────────────────────────────────────────

    #[test]
    fn audio_stream_roundtrip() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source(
            "audio_src",
            SourceConfig::File("audio.flac".into()),
            audio_spec(),
        );
        let sink = NodeSpec::sink(
            "audio_sink",
            SinkConfig::File("out.flac".into()),
            audio_spec(),
        );
        let s = g.add_node(src);
        let sk = g.add_node(sink);
        g.connect(s, "default", sk, "default").expect("connect");
        let g2 = roundtrip(&g);
        assert_eq!(g2.node_count(), 2);
        assert_eq!(g2.edge_count(), 1);
    }
}
