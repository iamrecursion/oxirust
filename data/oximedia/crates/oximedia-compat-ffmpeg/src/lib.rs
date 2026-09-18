//! FFmpeg CLI argument compatibility layer for OxiMedia.
//!
//! Parses FFmpeg-style command-line arguments and translates them
//! to OxiMedia `TranscodeConfig` and `FilterGraph` operations.
//!
//! ## Quick Start
//!
//! ```rust
//! use oximedia_compat_ffmpeg::parse_and_translate;
//!
//! let args: Vec<String> = vec![
//!     "-i".into(), "input.mkv".into(),
//!     "-c:v".into(), "libaom-av1".into(),
//!     "-crf".into(), "28".into(),
//!     "-c:a".into(), "libopus".into(),
//!     "output.webm".into(),
//! ];
//!
//! let result = parse_and_translate(&args);
//! for diag in &result.diagnostics {
//!     eprintln!("{}", diag.format_ffmpeg_style("oximedia-ff"));
//! }
//! ```
//!
//! ## Features
//!
//! ### `-filter_complex` parsing — `FilterGraph::parse()`
//!
//! Recursive-descent parser for the FFmpeg `-filter_complex` syntax. Produces an AST of
//! filter chains, pad labels, and per-filter options.
//!
//! ```rust
//! use oximedia_compat_ffmpeg::filter_complex::FilterGraph;
//!
//! let g = FilterGraph::parse("[0:v]scale=1920:1080[bg];[bg][1:v]overlay=x=10:y=10[out]")
//!     .expect("valid filter_complex");
//! assert_eq!(g.chains.len(), 2);
//! ```
//!
//! ### `-map` stream specifiers — `StreamSelector::parse()`
//!
//! Parses FFmpeg stream selector strings into typed `StreamSelector` values covering
//! file index, stream type, positional index, PID, metadata, and label selectors.
//!
//! ```rust
//! use oximedia_compat_ffmpeg::stream_spec::StreamSelector;
//!
//! // "0:v:0" — first video stream of input file 0
//! let sel = StreamSelector::parse("0:v:0").expect("valid specifier");
//!
//! // Negative prefix excludes a stream
//! let excl = StreamSelector::parse("-0:a:1").expect("valid negative specifier");
//!
//! // Filter-complex output pad label
//! let lbl = StreamSelector::parse("[out_v]").expect("valid label specifier");
//! ```
//!
//! ### `-ss` / `-to` / `-t` duration parsing — `parse_duration()`
//!
//! Parses FFmpeg duration strings in all common formats into `std::time::Duration`.
//! `check_seek_args()` validates that `-to` and `-t` are not both specified.
//!
//! ```rust
//! use oximedia_compat_ffmpeg::seek::parse_duration;
//! use std::time::Duration;
//!
//! assert_eq!(parse_duration("01:23:45.500").unwrap(), Duration::from_millis(5_025_500));
//! assert_eq!(parse_duration("2h").unwrap(),           Duration::from_secs(7_200));
//! assert_eq!(parse_duration("90m").unwrap(),          Duration::from_secs(5_400));
//! assert_eq!(parse_duration("300s").unwrap(),         Duration::from_secs(300));
//! ```
//!
//! ### `ffprobe -of json/xml/csv/default` — `format_probe_result()`
//!
//! Renders a `ProbeOutput` in any of the four standard ffprobe output formats.
//!
//! ```rust
//! use oximedia_compat_ffmpeg::ffprobe::{ProbeFormat, ProbeStream, ProbeOutput};
//! use oximedia_compat_ffmpeg::ffprobe_output::{FfprobeOutputFormat, format_probe_result};
//!
//! let stream = ProbeStream::new_video("av1", 1920, 1080, "16:9", 30.0);
//! let format = ProbeFormat::new("clip.mkv", "matroska,webm", 100_000_000, 60.0);
//! let output = ProbeOutput { format: Some(format), streams: vec![stream] };
//!
//! let json = format_probe_result(&output, FfprobeOutputFormat::Json).unwrap();
//! assert!(json.contains("\"codec_name\""));
//!
//! let xml = format_probe_result(&output, FfprobeOutputFormat::Xml).unwrap();
//! assert!(xml.contains("<ffprobe "));
//! ```
//!
//! ### APV codec aliases in the codec map
//!
//! The codec table recognises `"apv"` and `"apv1"` as direct matches for OxiMedia's
//! `CodecId::Apv` (ISO/IEC 23009-13 royalty-free intra-frame professional codec).
//! `"mjpeg"` and `"mjpegb"` similarly map directly to `CodecId::Mjpeg`.
//!
//! ```rust
//! use oximedia_compat_ffmpeg::codec_map::CodecMap;
//!
//! let map = CodecMap::new();
//! let entry = map.lookup("apv").expect("APV is a supported codec");
//! assert_eq!(entry.oxi_name, "apv");
//!
//! let entry2 = map.lookup("mjpeg").expect("MJPEG is a supported codec");
//! assert_eq!(entry2.oxi_name, "mjpeg");
//! ```
//!
//! ### Codec-map `OnceLock` caching
//!
//! The internal codec lookup table is built once on first access via a `std::sync::OnceLock`
//! and then shared for the lifetime of the process.  Subsequent calls to
//! [`codec_map::CodecMap::new`] or [`codec_map::CodecMap::lookup`] incur only a
//! pointer-load and hash-map lookup — no allocation or lock contention.
//!
//! ```
//! use oximedia_compat_ffmpeg::codec_map::CodecMap;
//!
//! // First call builds the table; all later calls reuse it.
//! let map1 = CodecMap::new();
//! let map2 = CodecMap::new();
//! let entry = map1.lookup("av1").expect("AV1 is supported");
//! assert_eq!(entry.oxi_name, "av1");
//! ```
//!
//! ### `-preset` / `-tune` / `-profile:v` — `EncoderQualityOptions`
//!
//! The argument parser recognises FFmpeg's three encoder quality flags and collects them
//! into [`EncoderQualityOptions`].  All three fields are `Option<_>`, so unset flags are
//! represented as `None`.
//!
//! | FFmpeg flag | Type | Values (examples) |
//! |-------------|------|-------------------|
//! | `-preset` | `EncoderQualityPreset` | `ultrafast` … `placebo` |
//! | `-tune` | `EncoderTune` | `film`, `animation`, `grain`, `zerolatency`, … |
//! | `-profile:v` | `EncoderProfile` | `baseline`, `main`, `high`, `high10`, … |
//!
//! ```
//! use oximedia_compat_ffmpeg::encoder_options::{EncoderQualityOptions, EncoderQualityPreset,
//!                                               EncoderTune, EncoderProfile};
//!
//! let opts = EncoderQualityOptions {
//!     preset:  Some("slow".parse().expect("valid preset")),
//!     tune:    Some("film".parse().expect("valid tune")),
//!     profile: Some("high".parse().expect("valid profile")),
//! };
//!
//! assert_eq!(opts.preset,  Some(EncoderQualityPreset::Slow));
//! assert_eq!(opts.tune,    Some(EncoderTune::Film));
//! assert_eq!(opts.profile, Some(EncoderProfile::High));
//! ```
//!
//! ### `-vf` / `-af` filter shorthand — `parse_vf` / `parse_af`
//!
//! [`parse_vf`] and [`parse_af`] parse the compact comma-separated filter syntax used with
//! FFmpeg's `-vf` and `-af` flags into a single-chain [`filter_complex::FilterGraph`].
//! Each filter entry may be `name` (no options) or `name=key:val:…`.
//!
//! ```
//! use oximedia_compat_ffmpeg::filter_shorthand::{parse_vf, parse_af};
//!
//! // Video filter: scale then format conversion
//! let vf = parse_vf("scale=1920:1080,format=yuv420p").expect("valid vf");
//! assert_eq!(vf.chains.len(), 1);
//! assert_eq!(vf.chains[0].filters.len(), 2);
//!
//! // Audio filter: resample
//! let af = parse_af("aresample=48000").expect("valid af");
//! assert_eq!(af.chains[0].filters[0].name, "aresample");
//! ```
//!
//! ### `-pass 1` / `-pass 2` two-pass encoding — `PassPhase`
//!
//! [`parse_pass`] extracts the `-pass` and optional `-passlogfile` flags from an argument
//! slice and returns a [`PassPhase`] enum that describes which pass should be performed
//! and where to read/write the statistics file.
//!
//! | Variant | Description |
//! |---------|-------------|
//! | `First { stats_path }` | Analysis pass — encode and collect statistics |
//! | `Second { stats_path }` | Quality pass — use statistics for optimal rate control |
//!
//! The default `stats_path` when `-passlogfile` is omitted is `ffmpeg2pass-0.log`.
//!
//! ```
//! use oximedia_compat_ffmpeg::pass::{parse_pass, PassPhase};
//! use std::path::PathBuf;
//!
//! let args: Vec<String> = vec![
//!     "-pass".into(), "1".into(),
//!     "-passlogfile".into(), "encode_stats".into(),
//! ];
//! let phase = parse_pass(&args).expect("no parse error").expect("phase present");
//! assert!(matches!(phase, PassPhase::First { stats_path } if stats_path == PathBuf::from("encode_stats")));
//!
//! let args2: Vec<String> = vec!["-pass".into(), "2".into()];
//! let phase2 = parse_pass(&args2).expect("no parse error").expect("phase present");
//! assert!(matches!(phase2, PassPhase::Second { .. }));
//! ```
//!
//! ## Codec and format mapping reference
//!
//! OxiMedia enforces a **patent-free-only** codec policy.  All patent-encumbered
//! codecs are automatically substituted with an equivalent royalty-free codec.
//! The table below lists the full mapping used by [`codec_map::CodecMap`].
//!
//! ### Video codecs
//!
//! | FFmpeg input name(s)                             | OxiMedia codec | Patent status        |
//! |--------------------------------------------------|----------------|----------------------|
//! | `av1`, `libaom-av1`, `libsvtav1`                | `av1`          | Royalty-free (direct)|
//! | `vp9`, `libvpx-vp9`                             | `vp9`          | Royalty-free (direct)|
//! | `vp8`, `libvpx`                                 | `vp8`          | Royalty-free (direct)|
//! | `ffv1`                                          | `ffv1`         | Royalty-free (direct)|
//! | `apv`, `apv1`                                   | `apv`          | Royalty-free (direct)|
//! | `mjpeg`, `mjpegb`                               | `mjpeg`        | Royalty-free (direct)|
//! | `libx264`, `h264`, `avc`                        | `av1`          | **Substituted** (H.264 → AV1)  |
//! | `libx265`, `hevc`, `h265`                       | `av1`          | **Substituted** (HEVC → AV1)   |
//! | `mpeg4`, `libxvid`, `xvid`                      | `av1`          | **Substituted** (MPEG-4 → AV1) |
//! | `mpeg2video`                                    | `av1`          | **Substituted** (MPEG-2 → AV1) |
//! | `wmv1`, `wmv2`, `wmv3`                          | `vp9`          | **Substituted** (WMV → VP9)    |
//! | `theora`, `libtheora`                           | `vp9`          | **Substituted** (Theora → VP9) |
//!
//! ### Audio codecs
//!
//! | FFmpeg input name(s)                             | OxiMedia codec | Patent status        |
//! |--------------------------------------------------|----------------|----------------------|
//! | `opus`, `libopus`                               | `opus`         | Royalty-free (direct)|
//! | `vorbis`, `libvorbis`                           | `vorbis`       | Royalty-free (direct)|
//! | `flac`                                          | `flac`         | Royalty-free (direct)|
//! | `pcm_s16le`, `pcm_s24le`, `pcm_f32le`, …       | `pcm`          | Royalty-free (direct)|
//! | `aac`, `libfaac`, `libfdk_aac`                  | `opus`         | **Substituted** (AAC → Opus)   |
//! | `mp3`, `libmp3lame`                             | `opus`         | **Substituted** (MP3 → Opus)   |
//! | `ac3`, `eac3`                                   | `flac`         | **Substituted** (AC-3 → FLAC)  |
//! | `dts`, `dtshd`                                  | `flac`         | **Substituted** (DTS → FLAC)   |
//! | `amr_nb`, `amr_wb`                              | `opus`         | **Substituted** (AMR → Opus)   |
//! | `wma`, `wmav2`                                  | `vorbis`       | **Substituted** (WMA → Vorbis) |
//!
//! ### Container / format mappings
//!
//! The `-f FORMAT` flag is passed through; OxiMedia uses the output file extension
//! as the primary muxer selector. The following aliases are normalised:
//!
//! | FFmpeg `-f` value            | OxiMedia container | Notes                           |
//! |------------------------------|--------------------|---------------------------------|
//! | `matroska`, `mkv`            | Matroska (`.mkv`)  | Primary AV1/VP9 container       |
//! | `webm`                       | WebM (`.webm`)     | VP8/VP9/AV1 + Opus/Vorbis       |
//! | `mp4`, `mov`, `m4v`          | MP4 (`.mp4`)       | MPEG-4 container (patent-free codecs only) |
//! | `ogg`, `oga`, `ogv`          | Ogg (`.ogg`)       | Ogg bitstream container         |
//! | `flac`                       | FLAC (`.flac`)     | Raw FLAC bitstream              |
//! | `wav`                        | WAVE (`.wav`)      | Uncompressed PCM                |
//! | `avi`                        | AVI (`.avi`)       | Legacy; VP8/VP9 supported       |
//!
//! ## Hardware acceleration (`-hwaccel`) — `hwaccel_compat`
//!
//! The [`hwaccel_compat`] module maps FFmpeg hardware backend names to OxiMedia's
//! `HwAccelConfig`. Since OxiMedia's GPU pipeline is software-first, hardware
//! acceleration is recorded but gracefully falls back to software when the
//! requested backend is unavailable.
//!
//! | FFmpeg `-hwaccel` value | [`HwBackend`] variant | Platform          |
//! |-------------------------|-----------------------|-------------------|
//! | `cuda`, `nvenc`, `cuvid`| `Cuda`                | NVIDIA            |
//! | `vaapi`                 | `Vaapi`               | Linux (VA-API)    |
//! | `qsv`                   | `QuickSync`           | Intel             |
//! | `amf`                   | `Amf`                 | AMD               |
//! | `videotoolbox`          | `VideoToolbox`        | macOS             |
//! | `vulkan`                | `Vulkan`              | Cross-platform    |
//! | `opencl`                | `OpenCl`              | Cross-platform    |
//! | `auto`                  | `Auto`                | Runtime detection |
//! | `none`, `software`      | `Software`            | No GPU            |
//!
//! ## Metadata mapping (`-metadata`, `-map_metadata`) — `metadata_compat`
//!
//! The [`metadata_compat`] module normalises FFmpeg tag keys to OxiMedia's
//! canonical lowercase key names. The `-map_metadata` flag is parsed into
//! [`MapMetadataDirective`] values that describe which input's tags to copy.
//!
//! | FFmpeg key(s)          | OxiMedia canonical key |
//! |------------------------|------------------------|
//! | `title`                | `title`                |
//! | `artist`, `author`     | `artist`               |
//! | `album`                | `album`                |
//! | `year`, `date`         | `year`                 |
//! | `track`, `tracknumber` | `tracknumber`          |
//! | `genre`                | `genre`                |
//! | `language`, `lang`     | `language`             |

pub mod arg_parser;
pub mod argument_builder;
pub mod batch_mode;
pub mod codec_map;
pub mod codec_mapping;
pub mod compat_ext;
pub mod concat_compat;
pub mod diagnostics;
pub mod encoder_options;
pub mod ffprobe;
pub mod ffprobe_output;
pub mod filter_complex;
pub mod filter_graph;
pub mod filter_lex;
pub mod filter_shorthand;
pub mod hwaccel_compat;
pub mod lavfi_compat;
pub mod metadata_compat;
pub mod pass;
pub mod preset_translator;
pub mod real_world_tests;
pub mod seek;
pub mod stream_spec;
pub mod translator;
pub mod two_pass;

pub use arg_parser::{
    FfmpegArgs, GlobalOptions, InputSpec, MapSpec, OutputSpec, StreamOptions, StreamType,
};
pub use argument_builder::FfmpegArgumentBuilder;
pub use codec_map::{CodecCategory, CodecEntry, CodecMap};
pub use codec_mapping::{CodecMapper, CodecMapping, FormatMapping};
pub use diagnostics::{Diagnostic, DiagnosticKind, DiagnosticSink, TranslationError};
pub use encoder_options::{
    EncoderProfile, EncoderQualityOptions, EncoderQualityPreset, EncoderTune,
};
pub use ffprobe_output::{format_probe_result, FfprobeOutputError, FfprobeOutputFormat};
pub use filter_complex::{
    Filter as FilterComplexFilter, FilterChain as FilterComplexChain, FilterComplexError,
    FilterGraph as FilterComplexGraph, FilterOption,
};
pub use filter_graph::{FilterChain, FilterGraph as AdvancedFilterGraph, FilterGraphNode};
pub use filter_lex::{
    parse_filter_graph, parse_filter_graph_zerocopy, parse_filter_string, parse_filters,
    FilterGraph, FilterNode, FilterToken, ParsedFilter,
};
pub use filter_shorthand::{parse_af, parse_vf};
pub use hwaccel_compat::{
    build_hw_accel_summary, parse_hwaccel_method, translate_hw_codec, translate_hwaccel,
    HwAccelConfig, HwAccelError, HwAccelSummary, HwBackend, HwCodecHint, HwCodecRole,
};
pub use metadata_compat::{
    extract_metadata_from_args, parse_metadata_arg, translate_metadata_args, MetadataError,
    MetadataMap, MetadataScope,
};
pub use pass::{parse_pass, PassPhase};
pub use seek::{check_seek_args, parse_duration, SeekError};
pub use stream_spec::{
    StreamIndex, StreamSelector, StreamSpec, StreamSpecError, StreamType as SpecStreamType,
};
pub use translator::{
    parse_and_translate, MapMetadataDirective, MuxerAction, MuxerOption, TranscodeJob,
    TranslateResult,
};
