//! Pre-defined pipeline templates for common media processing patterns.
//!
//! Templates encapsulate best-practice pipeline topologies for typical
//! use cases (transcoding, ABR ladder, thumbnail generation, audio normalisation,
//! picture-in-picture, and multi-output broadcast) so callers don't have to
//! assemble the graph by hand each time.
//!
//! Each template is parameterised via a [`TemplateConfig`] and materialised
//! via [`PipelineTemplate::build`].
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::templates::{PipelineTemplate, TemplateConfig};
//!
//! let config = TemplateConfig::transcode(
//!     "input.mp4",
//!     "output.mp4",
//!     1280, 720,
//! );
//! let graph = PipelineTemplate::Transcode
//!     .build(config)
//!     .expect("transcode template should build");
//!
//! assert!(graph.node_count() >= 3);
//! ```

use crate::graph::PipelineGraph;
use crate::node::{
    FilterConfig, FrameFormat, NodeSpec, NodeType, SinkConfig, SourceConfig, StreamSpec,
};
use crate::PipelineError;

// ── TemplateConfig ────────────────────────────────────────────────────────────

/// Configuration parameters for pipeline template instantiation.
///
/// Use the convenience constructors (`transcode`, `abr`, `thumbnail`, etc.) to
/// build the appropriate config without having to fill every field manually.
#[derive(Debug, Clone)]
pub struct TemplateConfig {
    /// Path (or URL) of the primary input media file.
    pub input_path: String,
    /// Primary output path or URL.
    pub output_path: String,
    /// Target video width in pixels (0 = keep source).
    pub target_width: u32,
    /// Target video height in pixels (0 = keep source).
    pub target_height: u32,
    /// Target video frame-rate (0 = keep source).
    pub target_fps: f32,
    /// Audio gain adjustment in dB (0.0 = unity).
    pub audio_gain_db: f32,
    /// For ABR templates: list of `(width, height, output_path)` renditions.
    pub abr_renditions: Vec<(u32, u32, String)>,
    /// For thumbnail template: output path for the thumbnail image.
    pub thumbnail_path: Option<String>,
    /// For thumbnail template: seek position in milliseconds.
    pub thumbnail_seek_ms: i64,
    /// For picture-in-picture: secondary input path.
    pub pip_secondary_path: Option<String>,
    /// For picture-in-picture: secondary stream position/size as `(x, y, w, h)`.
    pub pip_geometry: Option<(u32, u32, u32, u32)>,
    /// Optional trim window `(start_ms, end_ms)`. Both 0 means no trim.
    pub trim_window: Option<(i64, i64)>,
    /// Extra per-node key=value metadata injected into `Parametric` configs.
    pub extra_properties: Vec<(String, String)>,
}

impl TemplateConfig {
    /// Build a basic transcode configuration.
    pub fn transcode(
        input: impl Into<String>,
        output: impl Into<String>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            input_path: input.into(),
            output_path: output.into(),
            target_width: width,
            target_height: height,
            target_fps: 0.0,
            audio_gain_db: 0.0,
            abr_renditions: Vec::new(),
            thumbnail_path: None,
            thumbnail_seek_ms: 0,
            pip_secondary_path: None,
            pip_geometry: None,
            trim_window: None,
            extra_properties: Vec::new(),
        }
    }

    /// Build an ABR (Adaptive Bitrate) ladder configuration.
    pub fn abr(input: impl Into<String>, renditions: Vec<(u32, u32, String)>) -> Self {
        Self {
            input_path: input.into(),
            output_path: String::new(),
            target_width: 0,
            target_height: 0,
            target_fps: 0.0,
            audio_gain_db: 0.0,
            abr_renditions: renditions,
            thumbnail_path: None,
            thumbnail_seek_ms: 0,
            pip_secondary_path: None,
            pip_geometry: None,
            trim_window: None,
            extra_properties: Vec::new(),
        }
    }

    /// Build a thumbnail extraction configuration.
    pub fn thumbnail(
        input: impl Into<String>,
        output: impl Into<String>,
        seek_ms: i64,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            input_path: input.into(),
            output_path: String::new(),
            target_width: width,
            target_height: height,
            target_fps: 0.0,
            audio_gain_db: 0.0,
            abr_renditions: Vec::new(),
            thumbnail_path: Some(output.into()),
            thumbnail_seek_ms: seek_ms,
            pip_secondary_path: None,
            pip_geometry: None,
            trim_window: Some((seek_ms, seek_ms + 1)), // 1ms window
            extra_properties: Vec::new(),
        }
    }

    /// Build an audio normalisation pipeline configuration.
    pub fn audio_normalise(
        input: impl Into<String>,
        output: impl Into<String>,
        gain_db: f32,
    ) -> Self {
        Self {
            input_path: input.into(),
            output_path: output.into(),
            target_width: 0,
            target_height: 0,
            target_fps: 0.0,
            audio_gain_db: gain_db,
            abr_renditions: Vec::new(),
            thumbnail_path: None,
            thumbnail_seek_ms: 0,
            pip_secondary_path: None,
            pip_geometry: None,
            trim_window: None,
            extra_properties: Vec::new(),
        }
    }

    /// Build a picture-in-picture (PiP) overlay configuration.
    pub fn pip(
        input: impl Into<String>,
        secondary: impl Into<String>,
        output: impl Into<String>,
        pip_x: u32,
        pip_y: u32,
        pip_w: u32,
        pip_h: u32,
    ) -> Self {
        Self {
            input_path: input.into(),
            output_path: output.into(),
            target_width: 0,
            target_height: 0,
            target_fps: 0.0,
            audio_gain_db: 0.0,
            abr_renditions: Vec::new(),
            thumbnail_path: None,
            thumbnail_seek_ms: 0,
            pip_secondary_path: Some(secondary.into()),
            pip_geometry: Some((pip_x, pip_y, pip_w, pip_h)),
            trim_window: None,
            extra_properties: Vec::new(),
        }
    }

    /// Builder-style setter: apply a trim window to the config.
    pub fn with_trim(mut self, start_ms: i64, end_ms: i64) -> Self {
        self.trim_window = Some((start_ms, end_ms));
        self
    }

    /// Builder-style setter: set target frame rate.
    pub fn with_fps(mut self, fps: f32) -> Self {
        self.target_fps = fps;
        self
    }

    /// Builder-style setter: add an extra property applied to filter nodes.
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_properties.push((key.into(), value.into()));
        self
    }
}

// ── PipelineTemplate ──────────────────────────────────────────────────────────

/// Pre-defined pipeline topology variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineTemplate {
    /// Single-input, single-output transcoding pipeline with optional scale,
    /// trim, and fps conversion.
    Transcode,
    /// Multi-rendition ABR ladder: one input fanned out to N scaled outputs.
    AbrLadder,
    /// Extract a single thumbnail frame at a specific seek position.
    Thumbnail,
    /// Audio normalisation: adjust gain and write to output.
    AudioNormalise,
    /// Picture-in-picture overlay: composite a secondary stream onto the primary.
    PictureInPicture,
    /// Multi-output broadcast: duplicate the transcoded stream to two sinks
    /// (e.g. file + memory buffer for CDN upload).
    MultiOutputBroadcast,
}

impl PipelineTemplate {
    /// Instantiate the template with the given configuration and return a
    /// validated [`PipelineGraph`].
    ///
    /// Returns [`PipelineError::TemplateError`] if the config is missing
    /// required fields for the template variant.
    pub fn build(self, config: TemplateConfig) -> Result<PipelineGraph, PipelineError> {
        match self {
            PipelineTemplate::Transcode => build_transcode(config),
            PipelineTemplate::AbrLadder => build_abr_ladder(config),
            PipelineTemplate::Thumbnail => build_thumbnail(config),
            PipelineTemplate::AudioNormalise => build_audio_normalise(config),
            PipelineTemplate::PictureInPicture => build_pip(config),
            PipelineTemplate::MultiOutputBroadcast => build_multi_output(config),
        }
    }

    /// Return a human-readable description of this template.
    pub fn description(&self) -> &'static str {
        match self {
            PipelineTemplate::Transcode => {
                "Single-input → scale/trim/fps → single-output transcode"
            }
            PipelineTemplate::AbrLadder => {
                "Single-input → N scaled renditions (Adaptive Bitrate Ladder)"
            }
            PipelineTemplate::Thumbnail => {
                "Extract a thumbnail frame at a seek position, scale, and save"
            }
            PipelineTemplate::AudioNormalise => "Adjust audio gain and write normalised output",
            PipelineTemplate::PictureInPicture => {
                "Composite a secondary (PiP) stream onto the primary stream"
            }
            PipelineTemplate::MultiOutputBroadcast => {
                "Transcode and fan out to two sinks (file + memory buffer)"
            }
        }
    }
}

// ── Template builders ─────────────────────────────────────────────────────────

fn default_video_stream() -> StreamSpec {
    StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25)
}

fn default_audio_stream() -> StreamSpec {
    StreamSpec::audio(FrameFormat::S16Interleaved, 48000, 2)
}

/// Apply `extra_properties` from `config` to a `FilterConfig` via the
/// `with_property` builder chain.
fn apply_extra_props(mut filter: FilterConfig, config: &TemplateConfig) -> FilterConfig {
    for (k, v) in &config.extra_properties {
        filter = filter.with_property(k.clone(), v.clone());
    }
    filter
}

fn build_transcode(config: TemplateConfig) -> Result<PipelineGraph, PipelineError> {
    if config.input_path.is_empty() {
        return Err(PipelineError::TemplateError(
            "transcode: input_path is required".to_string(),
        ));
    }
    if config.output_path.is_empty() {
        return Err(PipelineError::TemplateError(
            "transcode: output_path is required".to_string(),
        ));
    }

    let mut g = PipelineGraph::new();
    let vs = default_video_stream();

    // Source
    let src = NodeSpec::source(
        "source",
        SourceConfig::File(config.input_path.clone()),
        vs.clone(),
    );
    let mut last_id = g.add_node(src);
    let mut last_pad = "default".to_string();

    // Optional trim
    if let Some((start_ms, end_ms)) = config.trim_window {
        if end_ms > start_ms {
            let trim_cfg = apply_extra_props(FilterConfig::Trim { start_ms, end_ms }, &config);
            let trim = NodeSpec::filter("trim", trim_cfg, vs.clone(), vs.clone());
            let trim_id = g.add_node(trim);
            g.connect(last_id, &last_pad, trim_id, "default")
                .map_err(|e| PipelineError::TemplateError(e.to_string()))?;
            last_id = trim_id;
            last_pad = "default".to_string();
        }
    }

    // Optional scale
    if config.target_width > 0 && config.target_height > 0 {
        let out_vs = StreamSpec::video(
            FrameFormat::Yuv420p,
            config.target_width,
            config.target_height,
            25,
        );
        let scale_cfg = apply_extra_props(
            FilterConfig::Scale {
                width: config.target_width,
                height: config.target_height,
            },
            &config,
        );
        let scale = NodeSpec::filter("scale", scale_cfg, vs.clone(), out_vs.clone());
        let scale_id = g.add_node(scale);
        g.connect(last_id, &last_pad, scale_id, "default")
            .map_err(|e| PipelineError::TemplateError(e.to_string()))?;
        last_id = scale_id;
        last_pad = "default".to_string();
    }

    // Optional fps
    if config.target_fps > 0.0 {
        let fps_cfg = apply_extra_props(
            FilterConfig::Fps {
                fps: config.target_fps,
            },
            &config,
        );
        let fps_node = NodeSpec::filter("fps", fps_cfg, vs.clone(), vs.clone());
        let fps_id = g.add_node(fps_node);
        g.connect(last_id, &last_pad, fps_id, "default")
            .map_err(|e| PipelineError::TemplateError(e.to_string()))?;
        last_id = fps_id;
        last_pad = "default".to_string();
    }

    // Sink
    let sink = NodeSpec::sink(
        "sink",
        SinkConfig::File(config.output_path.clone()),
        vs.clone(),
    );
    let sink_id = g.add_node(sink);
    g.connect(last_id, &last_pad, sink_id, "default")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

    Ok(g)
}

fn build_abr_ladder(config: TemplateConfig) -> Result<PipelineGraph, PipelineError> {
    if config.input_path.is_empty() {
        return Err(PipelineError::TemplateError(
            "abr: input_path is required".to_string(),
        ));
    }
    if config.abr_renditions.is_empty() {
        return Err(PipelineError::TemplateError(
            "abr: at least one rendition must be specified".to_string(),
        ));
    }

    let mut g = PipelineGraph::new();
    let vs = default_video_stream();

    let src = NodeSpec::source(
        "source",
        SourceConfig::File(config.input_path.clone()),
        vs.clone(),
    );
    let src_id = g.add_node(src);

    let n = config.abr_renditions.len();

    // Build a Split node with one output pad per rendition
    let output_pads: Vec<(String, StreamSpec)> =
        (0..n).map(|i| (format!("out{i}"), vs.clone())).collect();
    let split = NodeSpec::new(
        "split",
        NodeType::Split,
        vec![("default".to_string(), vs.clone())],
        output_pads,
    );
    let split_id = g.add_node(split);
    g.connect(src_id, "default", split_id, "default")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

    for (i, (w, h, out_path)) in config.abr_renditions.iter().enumerate() {
        let pad_name = format!("out{i}");
        let out_vs = StreamSpec::video(FrameFormat::Yuv420p, *w, *h, 25);
        let scale_cfg = apply_extra_props(
            FilterConfig::Scale {
                width: *w,
                height: *h,
            },
            &config,
        );
        let scale = NodeSpec::filter(
            &format!("scale_{w}x{h}"),
            scale_cfg,
            vs.clone(),
            out_vs.clone(),
        );
        let scale_id = g.add_node(scale);
        g.connect(split_id, &pad_name, scale_id, "default")
            .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

        let sink = NodeSpec::sink(
            &format!("sink_{w}x{h}"),
            SinkConfig::File(out_path.clone()),
            out_vs,
        );
        let sink_id = g.add_node(sink);
        g.connect(scale_id, "default", sink_id, "default")
            .map_err(|e| PipelineError::TemplateError(e.to_string()))?;
    }

    Ok(g)
}

fn build_thumbnail(config: TemplateConfig) -> Result<PipelineGraph, PipelineError> {
    if config.input_path.is_empty() {
        return Err(PipelineError::TemplateError(
            "thumbnail: input_path is required".to_string(),
        ));
    }
    let thumb_path = config
        .thumbnail_path
        .as_ref()
        .filter(|p| !p.is_empty())
        .ok_or_else(|| {
            PipelineError::TemplateError("thumbnail: thumbnail_path is required".to_string())
        })?
        .clone();

    let mut g = PipelineGraph::new();
    let vs = default_video_stream();

    let src = NodeSpec::source(
        "source",
        SourceConfig::File(config.input_path.clone()),
        vs.clone(),
    );
    let src_id = g.add_node(src);

    // Seek via Trim to extract a single frame
    let seek_ms = config.thumbnail_seek_ms;
    let trim_cfg = FilterConfig::Trim {
        start_ms: seek_ms,
        end_ms: seek_ms + 40, // ~one frame at 25fps
    };
    let trim = NodeSpec::filter("seek", trim_cfg, vs.clone(), vs.clone());
    let trim_id = g.add_node(trim);
    g.connect(src_id, "default", trim_id, "default")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

    let mut last_id = trim_id;

    // Optional scale
    if config.target_width > 0 && config.target_height > 0 {
        let out_vs = StreamSpec::video(
            FrameFormat::Yuv420p,
            config.target_width,
            config.target_height,
            25,
        );
        let scale_cfg = apply_extra_props(
            FilterConfig::Scale {
                width: config.target_width,
                height: config.target_height,
            },
            &config,
        );
        let scale = NodeSpec::filter("scale", scale_cfg, vs.clone(), out_vs.clone());
        let scale_id = g.add_node(scale);
        g.connect(last_id, "default", scale_id, "default")
            .map_err(|e| PipelineError::TemplateError(e.to_string()))?;
        last_id = scale_id;
    }

    let sink = NodeSpec::sink("thumb_sink", SinkConfig::File(thumb_path), vs.clone());
    let sink_id = g.add_node(sink);
    g.connect(last_id, "default", sink_id, "default")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

    Ok(g)
}

fn build_audio_normalise(config: TemplateConfig) -> Result<PipelineGraph, PipelineError> {
    if config.input_path.is_empty() {
        return Err(PipelineError::TemplateError(
            "audio_normalise: input_path is required".to_string(),
        ));
    }
    if config.output_path.is_empty() {
        return Err(PipelineError::TemplateError(
            "audio_normalise: output_path is required".to_string(),
        ));
    }

    let mut g = PipelineGraph::new();
    let aus = default_audio_stream();

    let src = NodeSpec::source(
        "source",
        SourceConfig::File(config.input_path.clone()),
        aus.clone(),
    );
    let src_id = g.add_node(src);

    let vol_cfg = apply_extra_props(
        FilterConfig::Volume {
            gain_db: config.audio_gain_db,
        },
        &config,
    );
    let vol = NodeSpec::filter("volume", vol_cfg, aus.clone(), aus.clone());
    let vol_id = g.add_node(vol);
    g.connect(src_id, "default", vol_id, "default")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

    let sink = NodeSpec::sink(
        "sink",
        SinkConfig::File(config.output_path.clone()),
        aus.clone(),
    );
    let sink_id = g.add_node(sink);
    g.connect(vol_id, "default", sink_id, "default")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

    Ok(g)
}

fn build_pip(config: TemplateConfig) -> Result<PipelineGraph, PipelineError> {
    if config.input_path.is_empty() {
        return Err(PipelineError::TemplateError(
            "pip: input_path is required".to_string(),
        ));
    }
    let secondary = config
        .pip_secondary_path
        .as_ref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            PipelineError::TemplateError("pip: pip_secondary_path is required".to_string())
        })?
        .clone();
    if config.output_path.is_empty() {
        return Err(PipelineError::TemplateError(
            "pip: output_path is required".to_string(),
        ));
    }
    let (_pip_x, _pip_y, pip_w, pip_h) = config
        .pip_geometry
        .ok_or_else(|| PipelineError::TemplateError("pip: pip_geometry is required".to_string()))?;

    let mut g = PipelineGraph::new();
    let vs = default_video_stream();

    // Primary source
    let primary_src = NodeSpec::source(
        "primary_src",
        SourceConfig::File(config.input_path.clone()),
        vs.clone(),
    );
    let primary_id = g.add_node(primary_src);

    // Secondary source
    let secondary_src =
        NodeSpec::source("secondary_src", SourceConfig::File(secondary), vs.clone());
    let secondary_id = g.add_node(secondary_src);

    // Scale secondary to PiP dimensions
    let pip_vs = StreamSpec::video(FrameFormat::Yuv420p, pip_w, pip_h, 25);
    let scale_cfg = apply_extra_props(
        FilterConfig::Scale {
            width: pip_w,
            height: pip_h,
        },
        &config,
    );
    let scale_pip = NodeSpec::filter("scale_pip", scale_cfg, vs.clone(), pip_vs);
    let scale_pip_id = g.add_node(scale_pip);
    g.connect(secondary_id, "default", scale_pip_id, "default")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

    // Crop the secondary at pip position in primary canvas
    // (x, y defines where PiP is placed; w, h already set by scale above)
    let pad_cfg = apply_extra_props(
        FilterConfig::Pad {
            width: 1920,
            height: 1080,
        },
        &config,
    );
    let pad_node = NodeSpec::filter("pad_pip", pad_cfg, vs.clone(), vs.clone());
    let pad_id = g.add_node(pad_node);
    g.connect(scale_pip_id, "default", pad_id, "default")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

    // Overlay merge node: two inputs (primary + padded pip)
    let merge = NodeSpec::new(
        "overlay",
        NodeType::Filter(apply_extra_props(FilterConfig::Overlay, &config)),
        vec![
            ("primary".to_string(), vs.clone()),
            ("overlay".to_string(), vs.clone()),
        ],
        vec![("default".to_string(), vs.clone())],
    );
    let merge_id = g.add_node(merge);
    g.connect(primary_id, "default", merge_id, "primary")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;
    g.connect(pad_id, "default", merge_id, "overlay")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

    // Sink
    let sink = NodeSpec::sink(
        "pip_sink",
        SinkConfig::File(config.output_path.clone()),
        vs.clone(),
    );
    let sink_id = g.add_node(sink);
    g.connect(merge_id, "default", sink_id, "default")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

    Ok(g)
}

fn build_multi_output(config: TemplateConfig) -> Result<PipelineGraph, PipelineError> {
    if config.input_path.is_empty() {
        return Err(PipelineError::TemplateError(
            "multi_output: input_path is required".to_string(),
        ));
    }
    if config.output_path.is_empty() {
        return Err(PipelineError::TemplateError(
            "multi_output: output_path (file sink) is required".to_string(),
        ));
    }

    let mut g = PipelineGraph::new();
    let vs = default_video_stream();

    // Source
    let src = NodeSpec::source(
        "source",
        SourceConfig::File(config.input_path.clone()),
        vs.clone(),
    );
    let src_id = g.add_node(src);

    // Optional scale
    let mut last_id = src_id;
    if config.target_width > 0 && config.target_height > 0 {
        let out_vs = StreamSpec::video(
            FrameFormat::Yuv420p,
            config.target_width,
            config.target_height,
            25,
        );
        let scale_cfg = apply_extra_props(
            FilterConfig::Scale {
                width: config.target_width,
                height: config.target_height,
            },
            &config,
        );
        let scale = NodeSpec::filter("scale", scale_cfg, vs.clone(), out_vs);
        let scale_id = g.add_node(scale);
        g.connect(last_id, "default", scale_id, "default")
            .map_err(|e| PipelineError::TemplateError(e.to_string()))?;
        last_id = scale_id;
    }

    // Split to two outputs
    let split = NodeSpec::new(
        "split",
        NodeType::Split,
        vec![("default".to_string(), vs.clone())],
        vec![
            ("out0".to_string(), vs.clone()),
            ("out1".to_string(), vs.clone()),
        ],
    );
    let split_id = g.add_node(split);
    g.connect(last_id, "default", split_id, "default")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

    // File sink
    let file_sink = NodeSpec::sink(
        "file_sink",
        SinkConfig::File(config.output_path.clone()),
        vs.clone(),
    );
    let file_sink_id = g.add_node(file_sink);
    g.connect(split_id, "out0", file_sink_id, "default")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

    // Memory buffer sink (for CDN upload / live preview)
    let mem_key = format!(
        "broadcast_{}",
        config.output_path.replace(['/', '\\', '.'], "_")
    );
    let mem_sink = NodeSpec::sink("buffer_sink", SinkConfig::Memory(mem_key), vs.clone());
    let mem_sink_id = g.add_node(mem_sink);
    g.connect(split_id, "out1", mem_sink_id, "default")
        .map_err(|e| PipelineError::TemplateError(e.to_string()))?;

    Ok(g)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::StreamKind;

    // ── Transcode ─────────────────────────────────────────────────────────────

    #[test]
    fn transcode_minimal() {
        let config = TemplateConfig::transcode("input.mp4", "output.mp4", 1280, 720);
        let graph = PipelineTemplate::Transcode
            .build(config)
            .expect("transcode should build");
        assert!(graph.node_count() >= 3); // src + scale + sink
        assert_eq!(graph.source_nodes().len(), 1);
        assert_eq!(graph.sink_nodes().len(), 1);
        let sorted = graph.topological_sort().expect("no cycle");
        assert!(!sorted.is_empty());
    }

    #[test]
    fn transcode_no_scale() {
        let config = TemplateConfig::transcode("in.mp4", "out.mp4", 0, 0);
        let graph = PipelineTemplate::Transcode.build(config).expect("build");
        // Only source + sink (no scale filter when width/height are 0)
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn transcode_with_trim() {
        let config =
            TemplateConfig::transcode("in.mp4", "out.mp4", 1280, 720).with_trim(1000, 5000);
        let graph = PipelineTemplate::Transcode.build(config).expect("build");
        // src + trim + scale + sink
        assert_eq!(graph.node_count(), 4);
    }

    #[test]
    fn transcode_with_fps() {
        let config = TemplateConfig::transcode("in.mp4", "out.mp4", 0, 0).with_fps(24.0);
        let graph = PipelineTemplate::Transcode.build(config).expect("build");
        // src + fps + sink
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn transcode_empty_input_error() {
        let config = TemplateConfig::transcode("", "out.mp4", 1280, 720);
        let result = PipelineTemplate::Transcode.build(config);
        assert!(matches!(result, Err(PipelineError::TemplateError(_))));
    }

    #[test]
    fn transcode_empty_output_error() {
        let config = TemplateConfig::transcode("in.mp4", "", 1280, 720);
        let result = PipelineTemplate::Transcode.build(config);
        assert!(matches!(result, Err(PipelineError::TemplateError(_))));
    }

    #[test]
    fn transcode_with_extra_properties() {
        let config = TemplateConfig::transcode("in.mp4", "out.mp4", 1280, 720)
            .with_property("preset", "fast")
            .with_property("crf", "23");
        let graph = PipelineTemplate::Transcode.build(config).expect("build");
        // scale node should have extra properties
        for spec in graph.nodes.values() {
            if let crate::node::NodeType::Filter(ref fc) = spec.node_type {
                if spec.name == "scale" {
                    assert_eq!(fc.get_property("preset"), Some("fast"));
                    assert_eq!(fc.get_property("crf"), Some("23"));
                }
            }
        }
    }

    // ── ABR ───────────────────────────────────────────────────────────────────

    #[test]
    fn abr_three_renditions() {
        let config = TemplateConfig::abr(
            "input.mp4",
            vec![
                (1920, 1080, "1080p.mp4".to_string()),
                (1280, 720, "720p.mp4".to_string()),
                (640, 360, "360p.mp4".to_string()),
            ],
        );
        let graph = PipelineTemplate::AbrLadder.build(config).expect("build");
        // src + split + 3*(scale+sink) = 1 + 1 + 6 = 8
        assert_eq!(graph.node_count(), 8);
        assert_eq!(graph.sink_nodes().len(), 3);
        assert_eq!(graph.source_nodes().len(), 1);
    }

    #[test]
    fn abr_empty_renditions_error() {
        let config = TemplateConfig::abr("input.mp4", vec![]);
        let result = PipelineTemplate::AbrLadder.build(config);
        assert!(matches!(result, Err(PipelineError::TemplateError(_))));
    }

    #[test]
    fn abr_empty_input_error() {
        let config = TemplateConfig::abr("", vec![(1280, 720, "out.mp4".to_string())]);
        let result = PipelineTemplate::AbrLadder.build(config);
        assert!(matches!(result, Err(PipelineError::TemplateError(_))));
    }

    #[test]
    fn abr_single_rendition() {
        let config = TemplateConfig::abr("in.mp4", vec![(640, 480, "sd.mp4".to_string())]);
        let graph = PipelineTemplate::AbrLadder.build(config).expect("build");
        // src + split + scale + sink = 4
        assert_eq!(graph.node_count(), 4);
    }

    // ── Thumbnail ─────────────────────────────────────────────────────────────

    #[test]
    fn thumbnail_with_scale() {
        let config = TemplateConfig::thumbnail("input.mp4", "thumb.jpg", 5000, 320, 180);
        let graph = PipelineTemplate::Thumbnail.build(config).expect("build");
        // src + seek(trim) + scale + sink = 4
        assert_eq!(graph.node_count(), 4);
    }

    #[test]
    fn thumbnail_no_scale() {
        let config = TemplateConfig::thumbnail("input.mp4", "thumb.jpg", 1000, 0, 0);
        let graph = PipelineTemplate::Thumbnail.build(config).expect("build");
        // src + seek(trim) + sink = 3
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn thumbnail_missing_output_error() {
        let config = TemplateConfig::thumbnail("input.mp4", "", 1000, 320, 180);
        let result = PipelineTemplate::Thumbnail.build(config);
        assert!(matches!(result, Err(PipelineError::TemplateError(_))));
    }

    // ── AudioNormalise ────────────────────────────────────────────────────────

    #[test]
    fn audio_normalise_basic() {
        let config = TemplateConfig::audio_normalise("in.wav", "out.wav", -6.0);
        let graph = PipelineTemplate::AudioNormalise
            .build(config)
            .expect("build");
        // src + vol + sink = 3
        assert_eq!(graph.node_count(), 3);
        // Verify stream kinds
        for edge in &graph.edges {
            let from_spec = &graph.nodes[&edge.from_node];
            let to_spec = &graph.nodes[&edge.to_node];
            // All pads should be audio
            for (_, ss) in &from_spec.output_pads {
                assert_eq!(ss.kind, StreamKind::Audio);
            }
            for (_, ss) in &to_spec.input_pads {
                assert_eq!(ss.kind, StreamKind::Audio);
            }
        }
    }

    #[test]
    fn audio_normalise_empty_input_error() {
        let config = TemplateConfig::audio_normalise("", "out.wav", 0.0);
        let result = PipelineTemplate::AudioNormalise.build(config);
        assert!(matches!(result, Err(PipelineError::TemplateError(_))));
    }

    // ── PictureInPicture ──────────────────────────────────────────────────────

    #[test]
    fn pip_basic() {
        let config = TemplateConfig::pip(
            "primary.mp4",
            "secondary.mp4",
            "output.mp4",
            100,
            100,
            320,
            180,
        );
        let graph = PipelineTemplate::PictureInPicture
            .build(config)
            .expect("build");
        // primary_src + secondary_src + scale_pip + pad_pip + overlay + pip_sink = 6
        assert_eq!(graph.node_count(), 6);
        assert_eq!(graph.source_nodes().len(), 2);
        assert_eq!(graph.sink_nodes().len(), 1);
    }

    #[test]
    fn pip_missing_secondary_error() {
        let config = TemplateConfig {
            input_path: "primary.mp4".to_string(),
            output_path: "out.mp4".to_string(),
            pip_secondary_path: None,
            pip_geometry: Some((0, 0, 320, 180)),
            ..TemplateConfig::transcode("x", "y", 0, 0)
        };
        let result = PipelineTemplate::PictureInPicture.build(config);
        assert!(matches!(result, Err(PipelineError::TemplateError(_))));
    }

    #[test]
    fn pip_missing_geometry_error() {
        let config = TemplateConfig {
            input_path: "primary.mp4".to_string(),
            output_path: "out.mp4".to_string(),
            pip_secondary_path: Some("secondary.mp4".to_string()),
            pip_geometry: None,
            ..TemplateConfig::transcode("x", "y", 0, 0)
        };
        let result = PipelineTemplate::PictureInPicture.build(config);
        assert!(matches!(result, Err(PipelineError::TemplateError(_))));
    }

    // ── MultiOutputBroadcast ──────────────────────────────────────────────────

    #[test]
    fn multi_output_basic() {
        let config = TemplateConfig::transcode("live.ts", "archive.mp4", 0, 0);
        let graph = PipelineTemplate::MultiOutputBroadcast
            .build(config)
            .expect("build");
        // src + split + file_sink + buffer_sink = 4
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.sink_nodes().len(), 2);
    }

    #[test]
    fn multi_output_with_scale() {
        let config = TemplateConfig::transcode("live.ts", "archive.mp4", 1280, 720);
        let graph = PipelineTemplate::MultiOutputBroadcast
            .build(config)
            .expect("build");
        // src + scale + split + file_sink + buffer_sink = 5
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.sink_nodes().len(), 2);
    }

    #[test]
    fn multi_output_empty_input_error() {
        let config = TemplateConfig::transcode("", "out.mp4", 0, 0);
        let result = PipelineTemplate::MultiOutputBroadcast.build(config);
        assert!(matches!(result, Err(PipelineError::TemplateError(_))));
    }

    // ── Template metadata ─────────────────────────────────────────────────────

    #[test]
    fn template_descriptions_are_nonempty() {
        let templates = [
            PipelineTemplate::Transcode,
            PipelineTemplate::AbrLadder,
            PipelineTemplate::Thumbnail,
            PipelineTemplate::AudioNormalise,
            PipelineTemplate::PictureInPicture,
            PipelineTemplate::MultiOutputBroadcast,
        ];
        for t in &templates {
            assert!(
                !t.description().is_empty(),
                "{t:?} description must not be empty"
            );
        }
    }

    #[test]
    fn all_templates_produce_valid_toposort() {
        let configs: Vec<(PipelineTemplate, TemplateConfig)> = vec![
            (
                PipelineTemplate::Transcode,
                TemplateConfig::transcode("in.mp4", "out.mp4", 1280, 720),
            ),
            (
                PipelineTemplate::AbrLadder,
                TemplateConfig::abr(
                    "in.mp4",
                    vec![
                        (1280, 720, "720p.mp4".to_string()),
                        (640, 360, "360p.mp4".to_string()),
                    ],
                ),
            ),
            (
                PipelineTemplate::Thumbnail,
                TemplateConfig::thumbnail("in.mp4", "th.jpg", 2000, 320, 180),
            ),
            (
                PipelineTemplate::AudioNormalise,
                TemplateConfig::audio_normalise("in.wav", "out.wav", -3.0),
            ),
            (
                PipelineTemplate::PictureInPicture,
                TemplateConfig::pip("p.mp4", "s.mp4", "o.mp4", 0, 0, 320, 180),
            ),
            (
                PipelineTemplate::MultiOutputBroadcast,
                TemplateConfig::transcode("live.ts", "archive.mp4", 0, 0),
            ),
        ];
        for (template, config) in configs {
            let graph = template.build(config).expect("template should build");
            let sorted = graph.topological_sort();
            assert!(
                sorted.is_ok(),
                "{template:?} should produce a DAG, got: {:?}",
                sorted.err()
            );
        }
    }
}
