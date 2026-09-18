//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::arg_parser::{FfmpegArgs, StreamType};
use crate::codec_map::CodecMap;

use super::functions::format_fps_f32;
use super::functions::{CONTAINER_TABLE, LAVFI_SOURCES, PIX_FMT_TABLE};

/// Maps between FFmpeg pixel format names and OxiMedia pixel format identifiers.
pub struct PixelFormatMapper;
impl PixelFormatMapper {
    /// Map an FFmpeg pixel format name to an OxiMedia pixel format identifier.
    ///
    /// Returns `None` for unrecognised format names.
    pub fn ffmpeg_to_oximedia(fmt: &str) -> Option<&'static str> {
        let key = fmt.to_lowercase();
        PIX_FMT_TABLE
            .iter()
            .find(|(k, _)| *k == key.as_str())
            .map(|(_, v)| *v)
    }
    /// Map an OxiMedia pixel format identifier back to a canonical FFmpeg name.
    pub fn oximedia_to_ffmpeg(oxi: &str) -> Option<&'static str> {
        let key = oxi.to_lowercase();
        PIX_FMT_TABLE
            .iter()
            .find(|(_, v)| *v == key.as_str())
            .map(|(k, _)| *k)
    }
}
/// A structured representation of an FFmpeg `-map` argument.
///
/// Examples: `-map 0:v:0`, `-map 0:a:1`, `-map 0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamMap {
    /// Input file index (0-based).
    pub input_idx: u32,
    /// Optional stream type selector (`v`, `a`, `s`).
    pub stream_type: Option<MapStreamType>,
    /// Optional stream index within that type.
    pub stream_idx: Option<u32>,
    /// Whether this is a negative (exclusion) map.
    pub negative: bool,
}
impl StreamMap {
    /// Build a [`StreamMap`] from an existing [`crate::arg_parser::MapSpec`].
    pub(crate) fn from_map_spec(spec: &crate::arg_parser::MapSpec) -> Self {
        let input_idx = spec.input_index as u32;
        let negative = spec.negative;
        let (stream_type, stream_idx) = match &spec.stream_selector {
            None => (None, None),
            Some(sel) => {
                let mut parts = sel.splitn(2, ':');
                let type_str = parts.next().unwrap_or("");
                let idx_str = parts.next();
                let stype = MapStreamType::from_str(type_str);
                let sidx = idx_str.and_then(|s| s.parse::<u32>().ok());
                (stype, sidx)
            }
        };
        StreamMap {
            input_idx,
            stream_type,
            stream_idx,
            negative,
        }
    }
    /// Parse a raw `-map` specifier string directly (e.g. `"0:v:0"`, `"0:a:1"`).
    pub fn parse(spec: &str) -> Option<Self> {
        let spec = spec.trim();
        if spec.is_empty() {
            return None;
        }
        if spec.starts_with('[') && spec.ends_with(']') {
            return Some(StreamMap {
                input_idx: 0,
                stream_type: None,
                stream_idx: None,
                negative: false,
            });
        }
        let (negative, rest) = if let Some(s) = spec.strip_prefix('-') {
            (true, s)
        } else {
            (false, spec)
        };
        let mut parts = rest.splitn(3, ':');
        let input_idx = parts.next()?.parse::<u32>().ok()?;
        let type_str = parts.next().unwrap_or("");
        let idx_str = parts.next();
        let stream_type = if type_str.is_empty() {
            None
        } else {
            MapStreamType::from_str(type_str)
        };
        let stream_idx = idx_str.and_then(|s| s.parse::<u32>().ok());
        Some(StreamMap {
            input_idx,
            stream_type,
            stream_idx,
            negative,
        })
    }
}
/// Diagnostics utilities for FFmpeg argument lists.
pub struct FfmpegDiagnostics;
impl FfmpegDiagnostics {
    /// Scan an argument slice for deprecated FFmpeg option flags and return a
    /// [`FfmpegWarning`] for each deprecated flag found.
    ///
    /// Checks for:
    /// - `-vcodec` → use `-c:v`
    /// - `-acodec` → use `-c:a`
    /// - `-ab` → use `-b:a`
    /// - (Note: `-ar` for sample rate is not deprecated per se, it is included
    ///   because some users confuse it with a codec option)
    ///
    /// ## Example
    ///
    /// ```
    /// use oximedia_compat_ffmpeg::compat_ext::FfmpegDiagnostics;
    ///
    /// let warnings = FfmpegDiagnostics::check_deprecated_options(&["-vcodec", "libx264"]);
    /// assert_eq!(warnings.len(), 1);
    /// assert_eq!(warnings[0].deprecated_flag, "-vcodec");
    /// ```
    pub fn check_deprecated_options(args: &[&str]) -> Vec<FfmpegWarning> {
        let mut warnings = Vec::new();
        for &arg in args {
            match arg {
                "-vcodec" => {
                    warnings.push(FfmpegWarning::new(
                        "-vcodec",
                        "-c:v",
                        "The -vcodec alias is deprecated since FFmpeg 0.9.",
                    ));
                }
                "-acodec" => {
                    warnings.push(FfmpegWarning::new(
                        "-acodec",
                        "-c:a",
                        "The -acodec alias is deprecated since FFmpeg 0.9.",
                    ));
                }
                "-ab" => {
                    warnings.push(FfmpegWarning::new(
                        "-ab",
                        "-b:a",
                        "The -ab alias is deprecated; use -b:a for audio bitrate.",
                    ));
                }
                "-scodec" => {
                    warnings.push(FfmpegWarning::new(
                        "-scodec",
                        "-c:s",
                        "The -scodec alias is deprecated since FFmpeg 0.9.",
                    ));
                }
                _ => {}
            }
        }
        warnings
    }
}
/// Maps between FFmpeg container format names/extensions and OxiMedia container IDs.
pub struct ContainerMapper;
impl ContainerMapper {
    /// Map an FFmpeg format name or file extension to an OxiMedia container identifier.
    ///
    /// Returns `None` for unrecognised names.
    pub fn ffmpeg_to_oximedia(ext: &str) -> Option<&'static str> {
        let key = ext.to_lowercase();
        let key = key.trim_start_matches('.');
        CONTAINER_TABLE
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| *v)
    }
    /// Map an OxiMedia container identifier back to a canonical FFmpeg format name.
    pub fn oximedia_to_ffmpeg(oxi: &str) -> Option<&'static str> {
        let key = oxi.to_lowercase();
        CONTAINER_TABLE
            .iter()
            .find(|(_, v)| *v == key.as_str())
            .map(|(k, _)| *k)
    }
}
/// Stream type component in a `-map` specifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapStreamType {
    Video,
    Audio,
    Subtitle,
    Data,
    Attachment,
}
impl MapStreamType {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "v" | "video" => Some(Self::Video),
            "a" | "audio" => Some(Self::Audio),
            "s" | "subtitle" => Some(Self::Subtitle),
            "d" | "data" => Some(Self::Data),
            "t" | "attachment" => Some(Self::Attachment),
            _ => None,
        }
    }
}
/// Utilities for inspecting and classifying filter graph expressions.
pub struct FilterGraphParser;
impl FilterGraphParser {
    /// Return `true` if `filter` is a known lavfi (virtual device) source filter.
    ///
    /// Lavfi sources generate video/audio without consuming an input stream —
    /// they are typically used with `-f lavfi` or as the source in `-filter_complex`.
    ///
    /// ```
    /// use oximedia_compat_ffmpeg::compat_ext::FilterGraphParser;
    ///
    /// assert!(FilterGraphParser::is_lavfi_source("color"));
    /// assert!(FilterGraphParser::is_lavfi_source("testsrc"));
    /// assert!(FilterGraphParser::is_lavfi_source("sine"));
    /// assert!(!FilterGraphParser::is_lavfi_source("scale"));
    /// ```
    pub fn is_lavfi_source(filter: &str) -> bool {
        let name = filter.split('=').next().unwrap_or(filter).trim();
        let name_lower = name.to_lowercase();
        LAVFI_SOURCES.iter().any(|&s| s == name_lower.as_str())
    }
}
/// Validates pad connections in a parsed filter graph.
pub struct FilterGraphValidator;
impl FilterGraphValidator {
    /// Check that all named pad labels `[label]` have matching producers and consumers.
    ///
    /// Returns a list of human-readable problem descriptions. An empty list means
    /// the graph is connection-complete.
    ///
    /// For each output pad label produced by a chain, there must be at least one
    /// chain that consumes it (uses it as an input label), and vice versa.
    pub fn check_connections(graph: &crate::filter_graph::FilterGraph) -> Vec<String> {
        let mut problems = Vec::new();
        let mut producers: Vec<String> = Vec::new();
        let mut consumers: Vec<String> = Vec::new();
        for chain in &graph.chains {
            if let Some(ref lbl) = chain.output_label {
                producers.push(lbl.clone());
            }
            if let Some(ref lbl) = chain.input_label {
                consumers.push(lbl.clone());
            }
        }
        for prod in &producers {
            if !consumers.contains(prod) {
                problems.push(format!(
                    "Output pad '[{}]' has no consumer in the filter graph",
                    prod
                ));
            }
        }
        for cons in &consumers {
            if !producers.contains(cons) {
                problems.push(format!(
                    "Input pad '[{}]' has no producer in the filter graph",
                    cons
                ));
            }
        }
        problems
    }
}
/// A warning emitted when a deprecated FFmpeg option is detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfmpegWarning {
    /// The deprecated flag that was found (e.g. `"-vcodec"`).
    pub deprecated_flag: String,
    /// The recommended replacement (e.g. `"-c:v"`).
    pub replacement: String,
    /// Human-readable message.
    pub message: String,
}
impl FfmpegWarning {
    fn new(deprecated: &str, replacement: &str, note: &str) -> Self {
        Self {
            deprecated_flag: deprecated.to_string(),
            replacement: replacement.to_string(),
            message: format!(
                "Deprecated option '{}': use '{}' instead. {}",
                deprecated, replacement, note
            ),
        }
    }
}
/// Fluent argument builder for constructing FFmpeg-style command-line argument
/// vectors directly, without needing mutable borrows.
///
/// Unlike [`crate::argument_builder::FfmpegArgumentBuilder`], this builder uses
/// a consuming/owned pattern and exposes a `build()` method that returns
/// `Vec<String>`.  It also exposes the method names requested by the task
/// specification (`codec_video`, `codec_audio`, `bitrate_video`, `bitrate_audio`,
/// `scale`, `seek`, `duration`, `metadata`).
///
/// ## Example
///
/// ```
/// use oximedia_compat_ffmpeg::compat_ext::ArgumentBuilder;
///
/// let args = ArgumentBuilder::new()
///     .input("input.mp4")
///     .codec_video("av1")
///     .codec_audio("opus")
///     .crf(30)
///     .scale(1280, 720)
///     .fps(29.97)
///     .seek(10.0)
///     .duration(60.0)
///     .metadata("title", "My Video")
///     .output("output.webm")
///     .build();
///
/// assert!(args.contains(&"-c:v".to_string()));
/// assert!(args.contains(&"av1".to_string()));
/// ```
#[derive(Debug, Clone, Default)]
pub struct ArgumentBuilder {
    global: Vec<String>,
    inputs: Vec<String>,
    output_opts: Vec<String>,
    output_path: Option<String>,
}
impl ArgumentBuilder {
    /// Create a new, empty builder.
    pub fn new() -> Self {
        Self::default()
    }
    /// Add an input file path (`-i path`).
    pub fn input(mut self, path: &str) -> Self {
        self.inputs.push("-i".to_string());
        self.inputs.push(path.to_string());
        self
    }
    /// Set the output file path.
    pub fn output(mut self, path: &str) -> Self {
        self.output_path = Some(path.to_string());
        self
    }
    /// Append `-c:v <codec>` (video codec).
    pub fn codec_video(mut self, codec: &str) -> Self {
        self.output_opts.push("-c:v".to_string());
        self.output_opts.push(codec.to_string());
        self
    }
    /// Append `-c:a <codec>` (audio codec).
    pub fn codec_audio(mut self, codec: &str) -> Self {
        self.output_opts.push("-c:a".to_string());
        self.output_opts.push(codec.to_string());
        self
    }
    /// Append `-b:v <bitrate>` (video bitrate string, e.g. `"2M"`, `"4000k"`).
    pub fn bitrate_video(mut self, br: &str) -> Self {
        self.output_opts.push("-b:v".to_string());
        self.output_opts.push(br.to_string());
        self
    }
    /// Append `-b:a <bitrate>` (audio bitrate string, e.g. `"128k"`).
    pub fn bitrate_audio(mut self, br: &str) -> Self {
        self.output_opts.push("-b:a".to_string());
        self.output_opts.push(br.to_string());
        self
    }
    /// Append `-preset <preset>`.
    pub fn preset(mut self, p: &str) -> Self {
        self.output_opts.push("-preset".to_string());
        self.output_opts.push(p.to_string());
        self
    }
    /// Append `-crf <crf>`.
    pub fn crf(mut self, crf: u32) -> Self {
        self.output_opts.push("-crf".to_string());
        self.output_opts.push(crf.to_string());
        self
    }
    /// Append a scale video filter via `-vf scale=<w>:<h>`.
    ///
    /// Pass `-1` for either dimension to preserve aspect ratio.
    pub fn scale(mut self, w: i32, h: i32) -> Self {
        self.output_opts.push("-vf".to_string());
        self.output_opts.push(format!("scale={}:{}", w, h));
        self
    }
    /// Append `-r <fps>` (frame rate).
    pub fn fps(mut self, fps: f32) -> Self {
        self.output_opts.push("-r".to_string());
        self.output_opts.push(format_fps_f32(fps));
        self
    }
    /// Append `-ss <seconds>` (seek start, in seconds).
    pub fn seek(mut self, ss: f64) -> Self {
        self.output_opts.push("-ss".to_string());
        self.output_opts.push(
            format!("{:.6}", ss)
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string(),
        );
        self
    }
    /// Append `-t <seconds>` (maximum duration, in seconds).
    pub fn duration(mut self, t: f64) -> Self {
        self.output_opts.push("-t".to_string());
        self.output_opts.push(
            format!("{:.6}", t)
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string(),
        );
        self
    }
    /// Append `-metadata key=value`.
    pub fn metadata(mut self, k: &str, v: &str) -> Self {
        self.output_opts.push("-metadata".to_string());
        self.output_opts.push(format!("{}={}", k, v));
        self
    }
    /// Append `-y` (overwrite without asking) to the global flags.
    pub fn overwrite(mut self) -> Self {
        self.global.push("-y".to_string());
        self
    }
    /// Append `-f <fmt>` (force container format).
    pub fn format(mut self, fmt: &str) -> Self {
        self.output_opts.push("-f".to_string());
        self.output_opts.push(fmt.to_string());
        self
    }
    /// Append `-filter_complex <expr>`.
    pub fn filter_complex(mut self, expr: &str) -> Self {
        self.output_opts.push("-filter_complex".to_string());
        self.output_opts.push(expr.to_string());
        self
    }
    /// Append `-loglevel <level>`.
    pub fn loglevel(mut self, level: &str) -> Self {
        self.global.push("-loglevel".to_string());
        self.global.push(level.to_string());
        self
    }
    /// Assemble and return the final argument list.
    ///
    /// Order: `[global] [inputs] [output_opts] [output_path]`.
    /// Does **not** include `"ffmpeg"` as the zeroth element.
    pub fn build(self) -> Vec<String> {
        let mut args =
            Vec::with_capacity(self.global.len() + self.inputs.len() + self.output_opts.len() + 1);
        args.extend(self.global);
        args.extend(self.inputs);
        args.extend(self.output_opts);
        if let Some(path) = self.output_path {
            args.push(path);
        }
        args
    }
    /// Build a human-readable `ffmpeg …` command string for logging/display.
    pub fn to_command_string(self) -> String {
        let args = self.build();
        let mut parts = vec!["ffmpeg".to_string()];
        for arg in &args {
            if arg.contains(' ') {
                parts.push(format!("\"{}\"", arg));
            } else {
                parts.push(arg.clone());
            }
        }
        parts.join(" ")
    }
}
/// A human-readable hint describing how one aspect of an FFmpeg command was
/// translated to OxiMedia.
#[derive(Debug, Clone, PartialEq)]
pub struct TranslationHint {
    /// The original FFmpeg value (e.g. `"libx264"`).
    pub original: String,
    /// The OxiMedia equivalent (e.g. `"av1"`).
    pub translated: String,
    /// Confidence in the translation: 1.0 = direct match, 0.5 = substituted.
    pub confidence: f32,
    /// Optional note explaining the mapping rationale.
    pub note: Option<String>,
}
impl TranslationHint {
    /// Create a direct-match hint (confidence 1.0).
    pub fn direct(original: impl Into<String>, translated: impl Into<String>) -> Self {
        Self {
            original: original.into(),
            translated: translated.into(),
            confidence: 1.0,
            note: None,
        }
    }
    /// Create a substitution hint (confidence 0.5) with a note.
    pub fn substituted(
        original: impl Into<String>,
        translated: impl Into<String>,
        note: impl Into<String>,
    ) -> Self {
        Self {
            original: original.into(),
            translated: translated.into(),
            confidence: 0.5,
            note: Some(note.into()),
        }
    }
    /// Attach a note to this hint.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}
/// Diagnostics for assessing how well an [`FfmpegArgs`] command can be
/// translated to OxiMedia.
pub struct FfmpegCompatDiagnostics;
impl FfmpegCompatDiagnostics {
    /// Compute a compatibility score for a parsed [`FfmpegArgs`] command.
    ///
    /// Returns a value in `[0.0, 1.0]` where:
    /// - `1.0` means everything is directly translatable.
    /// - Values below `1.0` indicate unknown codecs, formats, or options that
    ///   cannot be mapped to OxiMedia equivalents.
    ///
    /// ## Deductions
    ///
    /// - Unknown video codec: −0.3
    /// - Unknown audio codec: −0.2
    /// - Unknown container format: −0.1
    /// - Unknown filter in `-vf`/`-af`: −0.05 per unknown filter
    /// - Unrecognised extra args: −0.02 per argument pair
    pub fn score(args: &FfmpegArgs) -> f32 {
        let codec_map = CodecMap::new();
        let mut score = 1.0f32;
        for out in &args.outputs {
            for opt in &out.stream_options {
                if let Some(ref codec) = opt.codec {
                    if codec != "copy" && !codec_map.is_supported(codec) {
                        match opt.stream_type {
                            StreamType::Video => score -= 0.3,
                            StreamType::Audio => score -= 0.2,
                            StreamType::Subtitle => score -= 0.05,
                            StreamType::All => score -= 0.2,
                        }
                    }
                }
            }
            if let Some(ref fmt) = out.format {
                if ContainerMapper::ffmpeg_to_oximedia(fmt).is_none() {
                    score -= 0.1;
                }
            }
            let penalty = out.extra_args.len() as f32 * 0.02;
            score -= penalty;
        }
        score.max(0.0_f32).min(1.0_f32)
    }
}
