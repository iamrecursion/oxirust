//! `ffprobe`-compatible output mode.
//!
//! This module provides structured container/stream metadata representations
//! that can be serialised to the same JSON and CSV formats produced by
//! `ffprobe -show_format -show_streams -print_format json|csv`.
//!
//! ## Supported output formats
//!
//! | Format | Description |
//! |--------|-------------|
//! | JSON   | `-print_format json` (default) |
//! | CSV    | `-print_format csv` |
//! | Flat   | `-print_format flat` (key=value) |
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_compat_ffmpeg::ffprobe::{ProbeFormat, ProbeStream, ProbeOutput, PrintFormat};
//!
//! let stream = ProbeStream::new_video("h264", 1920, 1080, "16:9", 30.0);
//! let format = ProbeFormat::new("input.mp4", "mp4", 5_000_000, 120.0);
//! let output = ProbeOutput { format: Some(format), streams: vec![stream] };
//!
//! let json = output.to_print_format(PrintFormat::Json);
//! assert!(json.contains("\"codec_name\""));
//! ```

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Output format enum
// ─────────────────────────────────────────────────────────────────────────────

/// The output format for probe results (mirrors `-print_format` in ffprobe).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrintFormat {
    /// JSON output (default; matches `ffprobe -print_format json`).
    #[default]
    Json,
    /// CSV output (matches `ffprobe -print_format csv`).
    Csv,
    /// Flat key=value pairs (matches `ffprobe -print_format flat`).
    Flat,
}

impl PrintFormat {
    /// Parse a `-print_format` value string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "json" => Some(Self::Json),
            "csv" => Some(Self::Csv),
            "flat" | "default" => Some(Self::Flat),
            _ => None,
        }
    }
}

impl std::fmt::Display for PrintFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json => write!(f, "json"),
            Self::Csv => write!(f, "csv"),
            Self::Flat => write!(f, "flat"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Codec type
// ─────────────────────────────────────────────────────────────────────────────

/// The type of a media stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecType {
    /// Video stream.
    Video,
    /// Audio stream.
    Audio,
    /// Subtitle stream.
    Subtitle,
    /// Data stream.
    Data,
    /// Attachment stream.
    Attachment,
}

impl std::fmt::Display for CodecType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Video => write!(f, "video"),
            Self::Audio => write!(f, "audio"),
            Self::Subtitle => write!(f, "subtitle"),
            Self::Data => write!(f, "data"),
            Self::Attachment => write!(f, "attachment"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProbeStream
// ─────────────────────────────────────────────────────────────────────────────

/// A single stream entry in `ffprobe` output (mirrors one `streams` array element).
#[derive(Debug, Clone)]
pub struct ProbeStream {
    /// Zero-based stream index within the container.
    pub index: usize,
    /// Codec name (e.g. `"h264"`, `"aac"`, `"opus"`).
    pub codec_name: String,
    /// Long codec name (e.g. `"H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10"`).
    pub codec_long_name: String,
    /// Stream type (`video`, `audio`, etc.).
    pub codec_type: CodecType,
    /// Codec tag string (e.g. `"avc1"`, `"mp4a"`).
    pub codec_tag_string: String,
    /// Codec tag as hexadecimal (e.g. `"0x31637661"`).
    pub codec_tag: String,
    /// Video: frame width in pixels.
    pub width: Option<u32>,
    /// Video: frame height in pixels.
    pub height: Option<u32>,
    /// Video: coded width (may differ from display width).
    pub coded_width: Option<u32>,
    /// Video: coded height.
    pub coded_height: Option<u32>,
    /// Video: display aspect ratio (e.g. `"16:9"`).
    pub display_aspect_ratio: Option<String>,
    /// Video: pixel format (e.g. `"yuv420p"`).
    pub pix_fmt: Option<String>,
    /// Video: frames per second (e.g. `"30000/1001"`).
    pub r_frame_rate: Option<String>,
    /// Video: average frame rate.
    pub avg_frame_rate: Option<String>,
    /// Audio: sample rate in Hz.
    pub sample_rate: Option<u32>,
    /// Audio: number of channels.
    pub channels: Option<u8>,
    /// Audio: channel layout string (e.g. `"stereo"`, `"5.1"`).
    pub channel_layout: Option<String>,
    /// Audio: sample format (e.g. `"fltp"`, `"s16"`).
    pub sample_fmt: Option<String>,
    /// Audio: bits per raw sample.
    pub bits_per_raw_sample: Option<u32>,
    /// Bits per second for this stream.
    pub bit_rate: Option<u64>,
    /// Time base as a fraction string (e.g. `"1/90000"`).
    pub time_base: Option<String>,
    /// Stream duration in time-base units.
    pub duration_ts: Option<i64>,
    /// Stream duration in seconds as a string.
    pub duration: Option<String>,
    /// Extra tags (title, language, etc.).
    pub tags: HashMap<String, String>,
    /// Video profile (e.g. `"High"`, `"Main"`).
    pub profile: Option<String>,
    /// Video level (e.g. `40` = 4.0).
    pub level: Option<i32>,
    /// Video: whether there are B-frames.
    pub has_b_frames: Option<u8>,
    /// Video: reference frames.
    pub refs: Option<u32>,
}

impl ProbeStream {
    /// Create a new video stream descriptor.
    pub fn new_video(
        codec_name: &str,
        width: u32,
        height: u32,
        aspect_ratio: &str,
        fps: f64,
    ) -> Self {
        let fps_str = format_fps_ratio(fps);
        Self {
            index: 0,
            codec_name: codec_name.to_string(),
            codec_long_name: codec_long_name(codec_name),
            codec_type: CodecType::Video,
            codec_tag_string: codec_tag_for(codec_name),
            codec_tag: "0x00000000".to_string(),
            width: Some(width),
            height: Some(height),
            coded_width: Some(width),
            coded_height: Some(height),
            display_aspect_ratio: Some(aspect_ratio.to_string()),
            pix_fmt: Some("yuv420p".to_string()),
            r_frame_rate: Some(fps_str.clone()),
            avg_frame_rate: Some(fps_str),
            sample_rate: None,
            channels: None,
            channel_layout: None,
            sample_fmt: None,
            bits_per_raw_sample: None,
            bit_rate: None,
            time_base: Some("1/90000".to_string()),
            duration_ts: None,
            duration: None,
            tags: HashMap::new(),
            profile: None,
            level: None,
            has_b_frames: Some(0),
            refs: Some(1),
        }
    }

    /// Create a new audio stream descriptor.
    pub fn new_audio(
        codec_name: &str,
        sample_rate: u32,
        channels: u8,
        channel_layout: &str,
    ) -> Self {
        Self {
            index: 0,
            codec_name: codec_name.to_string(),
            codec_long_name: codec_long_name(codec_name),
            codec_type: CodecType::Audio,
            codec_tag_string: codec_tag_for(codec_name),
            codec_tag: "0x00000000".to_string(),
            width: None,
            height: None,
            coded_width: None,
            coded_height: None,
            display_aspect_ratio: None,
            pix_fmt: None,
            r_frame_rate: None,
            avg_frame_rate: None,
            sample_rate: Some(sample_rate),
            channels: Some(channels),
            channel_layout: Some(channel_layout.to_string()),
            sample_fmt: Some("fltp".to_string()),
            bits_per_raw_sample: None,
            bit_rate: None,
            time_base: Some("1/48000".to_string()),
            duration_ts: None,
            duration: None,
            tags: HashMap::new(),
            profile: None,
            level: None,
            has_b_frames: None,
            refs: None,
        }
    }

    /// Render to JSON format (one `{}` object, no surrounding array brackets).
    pub fn to_json(&self) -> String {
        let mut fields: Vec<String> = Vec::new();
        fields.push(json_kv("index", &self.index.to_string()));
        fields.push(json_str("codec_name", &self.codec_name));
        fields.push(json_str("codec_long_name", &self.codec_long_name));
        fields.push(json_str("codec_type", &self.codec_type.to_string()));
        fields.push(json_str("codec_tag_string", &self.codec_tag_string));
        fields.push(json_str("codec_tag", &self.codec_tag));

        if let Some(v) = self.width {
            fields.push(json_kv("width", &v.to_string()));
        }
        if let Some(v) = self.height {
            fields.push(json_kv("height", &v.to_string()));
        }
        if let Some(v) = self.coded_width {
            fields.push(json_kv("coded_width", &v.to_string()));
        }
        if let Some(v) = self.coded_height {
            fields.push(json_kv("coded_height", &v.to_string()));
        }
        if let Some(v) = &self.display_aspect_ratio {
            fields.push(json_str("display_aspect_ratio", v));
        }
        if let Some(v) = &self.pix_fmt {
            fields.push(json_str("pix_fmt", v));
        }
        if let Some(v) = &self.r_frame_rate {
            fields.push(json_str("r_frame_rate", v));
        }
        if let Some(v) = &self.avg_frame_rate {
            fields.push(json_str("avg_frame_rate", v));
        }
        if let Some(v) = self.sample_rate {
            fields.push(json_str("sample_rate", &v.to_string()));
        }
        if let Some(v) = self.channels {
            fields.push(json_kv("channels", &v.to_string()));
        }
        if let Some(v) = &self.channel_layout {
            fields.push(json_str("channel_layout", v));
        }
        if let Some(v) = &self.sample_fmt {
            fields.push(json_str("sample_fmt", v));
        }
        if let Some(v) = self.bit_rate {
            fields.push(json_str("bit_rate", &v.to_string()));
        }
        if let Some(v) = &self.time_base {
            fields.push(json_str("time_base", v));
        }
        if let Some(v) = self.duration_ts {
            fields.push(json_kv("duration_ts", &v.to_string()));
        }
        if let Some(v) = &self.duration {
            fields.push(json_str("duration", v));
        }
        if let Some(v) = &self.profile {
            fields.push(json_str("profile", v));
        }
        if let Some(v) = self.level {
            fields.push(json_kv("level", &v.to_string()));
        }

        if !self.tags.is_empty() {
            let tag_fields: Vec<String> = self.tags.iter().map(|(k, v)| json_str(k, v)).collect();
            fields.push(format!(
                "        \"tags\": {{\n            {}\n        }}",
                tag_fields.join(",\n            ")
            ));
        }

        format!(
            "        {{\n            {}\n        }}",
            fields.join(",\n            ")
        )
    }

    /// Render to CSV format.
    ///
    /// Format: `stream,<index>,<codec_name>,<codec_type>,<width_or_blank>,<height_or_blank>,<sample_rate_or_blank>`
    pub fn to_csv(&self) -> String {
        format!(
            "stream,{},{},{},{},{},{}",
            self.index,
            csv_escape(&self.codec_name),
            csv_escape(&self.codec_type.to_string()),
            self.width.map(|v| v.to_string()).unwrap_or_default(),
            self.height.map(|v| v.to_string()).unwrap_or_default(),
            self.sample_rate.map(|v| v.to_string()).unwrap_or_default(),
        )
    }

    /// Render to flat key=value format.
    pub fn to_flat(&self, prefix: &str) -> String {
        let mut lines: Vec<String> = Vec::new();
        let idx = self.index;
        lines.push(format!(
            "{}streams.stream.{}.index={}",
            prefix, idx, self.index
        ));
        lines.push(format!(
            "{}streams.stream.{}.codec_name={}",
            prefix, idx, self.codec_name
        ));
        lines.push(format!(
            "{}streams.stream.{}.codec_type={}",
            prefix, idx, self.codec_type
        ));
        if let Some(w) = self.width {
            lines.push(format!("{}streams.stream.{}.width={}", prefix, idx, w));
        }
        if let Some(h) = self.height {
            lines.push(format!("{}streams.stream.{}.height={}", prefix, idx, h));
        }
        if let Some(sr) = self.sample_rate {
            lines.push(format!(
                "{}streams.stream.{}.sample_rate={}",
                prefix, idx, sr
            ));
        }
        if let Some(ch) = self.channels {
            lines.push(format!("{}streams.stream.{}.channels={}", prefix, idx, ch));
        }
        lines.join("\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProbeFormat
// ─────────────────────────────────────────────────────────────────────────────

/// Container-level information (mirrors the `format` object in `ffprobe` output).
#[derive(Debug, Clone)]
pub struct ProbeFormat {
    /// Input filename.
    pub filename: String,
    /// Number of streams.
    pub nb_streams: usize,
    /// Number of programs.
    pub nb_programs: usize,
    /// Format name (e.g. `"mov,mp4,m4a,3gp,3g2,mj2"`).
    pub format_name: String,
    /// Long format name.
    pub format_long_name: String,
    /// Start time in seconds.
    pub start_time: Option<f64>,
    /// Total duration in seconds.
    pub duration: Option<f64>,
    /// File size in bytes.
    pub size: Option<u64>,
    /// Overall bit rate in bits/s.
    pub bit_rate: Option<u64>,
    /// Probe score (0–100).
    pub probe_score: u8,
    /// Metadata tags (title, artist, encoder, etc.).
    pub tags: HashMap<String, String>,
}

impl ProbeFormat {
    /// Create a new format descriptor.
    pub fn new(filename: &str, format_name: &str, size: u64, duration_secs: f64) -> Self {
        let long_name = format_long_name(format_name);
        Self {
            filename: filename.to_string(),
            nb_streams: 0,
            nb_programs: 0,
            format_name: format_name.to_string(),
            format_long_name: long_name,
            start_time: Some(0.0),
            duration: Some(duration_secs),
            size: Some(size),
            bit_rate: size.checked_mul(8).and_then(|bits| {
                if duration_secs > 0.0 {
                    Some((bits as f64 / duration_secs) as u64)
                } else {
                    None
                }
            }),
            probe_score: 100,
            tags: HashMap::new(),
        }
    }

    /// Render to JSON format.
    pub fn to_json(&self) -> String {
        let mut fields: Vec<String> = Vec::new();
        fields.push(json_str("filename", &self.filename));
        fields.push(json_kv("nb_streams", &self.nb_streams.to_string()));
        fields.push(json_kv("nb_programs", &self.nb_programs.to_string()));
        fields.push(json_str("format_name", &self.format_name));
        fields.push(json_str("format_long_name", &self.format_long_name));
        if let Some(v) = self.start_time {
            fields.push(json_str("start_time", &format!("{:.6}", v)));
        }
        if let Some(v) = self.duration {
            fields.push(json_str("duration", &format!("{:.6}", v)));
        }
        if let Some(v) = self.size {
            fields.push(json_str("size", &v.to_string()));
        }
        if let Some(v) = self.bit_rate {
            fields.push(json_str("bit_rate", &v.to_string()));
        }
        fields.push(json_kv("probe_score", &self.probe_score.to_string()));

        if !self.tags.is_empty() {
            let tag_fields: Vec<String> = self.tags.iter().map(|(k, v)| json_str(k, v)).collect();
            fields.push(format!(
                "        \"tags\": {{\n            {}\n        }}",
                tag_fields.join(",\n            ")
            ));
        }

        format!("    {{\n        {}\n    }}", fields.join(",\n        "))
    }

    /// Render to CSV format.
    pub fn to_csv(&self) -> String {
        format!(
            "format,{},{},{},{}",
            csv_escape(&self.filename),
            csv_escape(&self.format_name),
            self.size.map(|v| v.to_string()).unwrap_or_default(),
            self.duration
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
        )
    }

    /// Render to flat key=value format.
    pub fn to_flat(&self, prefix: &str) -> String {
        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("{}format.filename={}", prefix, self.filename));
        lines.push(format!("{}format.nb_streams={}", prefix, self.nb_streams));
        lines.push(format!("{}format.format_name={}", prefix, self.format_name));
        if let Some(v) = self.duration {
            lines.push(format!("{}format.duration={:.6}", prefix, v));
        }
        if let Some(v) = self.size {
            lines.push(format!("{}format.size={}", prefix, v));
        }
        if let Some(v) = self.bit_rate {
            lines.push(format!("{}format.bit_rate={}", prefix, v));
        }
        lines.join("\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProbeOutput
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level `ffprobe` output — combines format info and a list of streams.
#[derive(Debug, Clone, Default)]
pub struct ProbeOutput {
    /// Container format information, if `show_format` was requested.
    pub format: Option<ProbeFormat>,
    /// Stream information, if `show_streams` was requested.
    pub streams: Vec<ProbeStream>,
}

impl ProbeOutput {
    /// Create an empty probe output.
    pub fn new() -> Self {
        Self::default()
    }

    /// Render the entire probe output to the requested [`PrintFormat`].
    pub fn to_print_format(&self, fmt: PrintFormat) -> String {
        match fmt {
            PrintFormat::Json => self.to_json(),
            PrintFormat::Csv => self.to_csv(),
            PrintFormat::Flat => self.to_flat(),
        }
    }

    /// Render to JSON (the `ffprobe -print_format json` format).
    pub fn to_json(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        if !self.streams.is_empty() {
            let mut stream_parts: Vec<String> = self
                .streams
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let mut s = s.clone();
                    s.index = i;
                    s.to_json()
                })
                .collect();
            // Separate stream entries with commas.
            for i in 0..stream_parts.len().saturating_sub(1) {
                let entry = stream_parts[i].clone();
                stream_parts[i] = entry;
            }
            parts.push(format!(
                "    \"streams\": [\n{}\n    ]",
                stream_parts.join(",\n")
            ));
        }

        if let Some(ref fmt) = self.format {
            parts.push(format!("    \"format\": {}", fmt.to_json()));
        }

        format!("{{\n{}\n}}", parts.join(",\n"))
    }

    /// Render to CSV (the `ffprobe -print_format csv` format).
    pub fn to_csv(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        for (i, s) in self.streams.iter().enumerate() {
            let mut s = s.clone();
            s.index = i;
            lines.push(s.to_csv());
        }
        if let Some(ref fmt) = self.format {
            lines.push(fmt.to_csv());
        }
        lines.join("\n")
    }

    /// Render to flat key=value format.
    pub fn to_flat(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        for (i, s) in self.streams.iter().enumerate() {
            let mut s = s.clone();
            s.index = i;
            lines.push(s.to_flat(""));
        }
        if let Some(ref fmt) = self.format {
            lines.push(fmt.to_flat(""));
        }
        lines.join("\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn json_kv(key: &str, value: &str) -> String {
    format!("\"{}\": {}", key, value)
}

fn json_str(key: &str, value: &str) -> String {
    format!("\"{}\": \"{}\"", key, json_escape(value))
}

/// Minimal JSON string escaping.
fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Minimal CSV escaping (quote fields containing commas or quotes).
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Map an fps float to an `ffprobe`-style rational string.
fn format_fps_ratio(fps: f64) -> String {
    // NTSC special cases.
    if (fps - 29.97).abs() < 0.01 {
        return "30000/1001".to_string();
    }
    if (fps - 23.976).abs() < 0.01 {
        return "24000/1001".to_string();
    }
    if (fps - 59.94).abs() < 0.01 {
        return "60000/1001".to_string();
    }
    // Integer frame rates.
    let n = fps.round() as u32;
    if (fps - n as f64).abs() < 0.001 {
        return format!("{}/1", n);
    }
    // Generic rational approximation.
    let num = (fps * 1001.0).round() as u32;
    format!("{}/1001", num)
}

/// Return a human-readable long name for well-known codec identifiers.
fn codec_long_name(codec: &str) -> String {
    match codec.to_lowercase().as_str() {
        "h264" | "libx264" => "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
        "hevc" | "libx265" => "H.265 / HEVC (High Efficiency Video Coding)",
        "av1" | "libaom-av1" | "libsvtav1" => "Alliance for Open Media AV1",
        "vp9" | "libvpx-vp9" => "Google VP9",
        "vp8" | "libvpx" => "On2 VP8",
        "aac" => "AAC (Advanced Audio Coding)",
        "opus" | "libopus" => "Opus (Opus Interactive Audio Codec)",
        "vorbis" | "libvorbis" => "Vorbis",
        "flac" => "FLAC (Free Lossless Audio Codec)",
        "mp3" | "libmp3lame" => "MP3 (MPEG audio layer 3)",
        "ffv1" => "FFmpeg video codec #1",
        _ => codec,
    }
    .to_string()
}

/// Return a codec tag string for well-known codecs.
fn codec_tag_for(codec: &str) -> String {
    match codec.to_lowercase().as_str() {
        "h264" | "libx264" => "avc1",
        "hevc" | "libx265" => "hev1",
        "av1" | "libaom-av1" => "av01",
        "vp9" | "libvpx-vp9" => "vp09",
        "vp8" | "libvpx" => "vp08",
        "aac" => "mp4a",
        "opus" | "libopus" => "Opus",
        "vorbis" | "libvorbis" => "vorb",
        "flac" => "fLaC",
        "mp3" | "libmp3lame" => "mp3 ",
        _ => "0x0000",
    }
    .to_string()
}

/// Return a human-readable long format name.
fn format_long_name(fmt: &str) -> String {
    match fmt.to_lowercase().as_str() {
        "mp4" | "mov" => "QuickTime / MOV",
        "mkv" | "matroska" => "Matroska / WebM",
        "webm" => "WebM",
        "ogg" => "Ogg",
        "avi" => "AVI (Audio Video Interleaved)",
        "flv" => "FLV (Flash Video)",
        "ts" | "mpegts" => "MPEG-TS (MPEG-2 Transport Stream)",
        "flac" => "raw FLAC",
        "wav" => "WAV / WAVE (Waveform Audio)",
        "mp3" => "MP2/3 (MPEG audio layer 2/3)",
        _ => fmt,
    }
    .to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// FfprobeQuery — parsed ffprobe argument set
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed set of `ffprobe` query options derived from a command-line argument
/// slice (e.g. `ffprobe -v quiet -print_format json -show_format -show_streams`).
#[derive(Debug, Clone, PartialEq)]
pub struct FfprobeQuery {
    /// The input file path to probe.
    pub input_path: Option<String>,
    /// The requested output format (`json`, `csv`, `flat`, …).
    pub print_format: PrintFormat,
    /// Whether container-level format information was requested (`-show_format`).
    pub show_format: bool,
    /// Whether per-stream information was requested (`-show_streams`).
    pub show_streams: bool,
    /// Whether per-packet information was requested (`-show_packets`).
    pub show_packets: bool,
    /// Verbosity level override (`-v quiet|info|verbose|…`).
    pub verbosity: Option<String>,
    /// Selected stream specifier (`-select_streams v:0`, `a`, …).
    pub select_streams: Option<String>,
}

impl Default for FfprobeQuery {
    fn default() -> Self {
        Self {
            input_path: None,
            print_format: PrintFormat::Json,
            show_format: false,
            show_streams: false,
            show_packets: false,
            verbosity: None,
            select_streams: None,
        }
    }
}

/// Error returned when an `ffprobe` argument slice cannot be translated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FfprobeArgError {
    /// A required value was missing after a flag.
    MissingValue(String),
    /// An unrecognised `-print_format` value was supplied.
    UnknownPrintFormat(String),
}

impl std::fmt::Display for FfprobeArgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingValue(flag) => write!(f, "missing value after '{}'", flag),
            Self::UnknownPrintFormat(v) => {
                write!(f, "unrecognised -print_format value '{}'", v)
            }
        }
    }
}

impl std::error::Error for FfprobeArgError {}

/// Translate a raw `ffprobe` argument slice into a [`FfprobeQuery`].
///
/// Recognised flags:
/// - `-i <path>` / positional path — input file
/// - `-print_format <json|csv|flat|default>` / `-of <format>` — output format
/// - `-show_format` — include container metadata
/// - `-show_streams` — include per-stream metadata
/// - `-show_packets` — include per-packet metadata
/// - `-v <level>` / `-loglevel <level>` — verbosity
/// - `-select_streams <spec>` — filter streams
///
/// Returns an error only for strictly invalid argument combinations (e.g.
/// missing value after a flag that requires one).
///
/// # Example
///
/// ```rust
/// use oximedia_compat_ffmpeg::ffprobe::{translate_ffprobe_args, PrintFormat};
///
/// let args = &["-v", "quiet", "-print_format", "json", "-show_format",
///              "-show_streams", "-i", "movie.mkv"];
/// let q = translate_ffprobe_args(args).unwrap();
/// assert_eq!(q.print_format, PrintFormat::Json);
/// assert!(q.show_format);
/// assert!(q.show_streams);
/// assert_eq!(q.input_path.as_deref(), Some("movie.mkv"));
/// ```
pub fn translate_ffprobe_args(args: &[&str]) -> Result<FfprobeQuery, FfprobeArgError> {
    let mut query = FfprobeQuery::default();
    let mut it = args.iter().copied().peekable();

    while let Some(arg) = it.next() {
        match arg {
            "-i" => {
                let path = it
                    .next()
                    .ok_or_else(|| FfprobeArgError::MissingValue("-i".to_string()))?;
                query.input_path = Some(path.to_string());
            }
            "-print_format" | "-of" => {
                let val = it
                    .next()
                    .ok_or_else(|| FfprobeArgError::MissingValue(arg.to_string()))?;
                query.print_format = PrintFormat::from_str(val)
                    .ok_or_else(|| FfprobeArgError::UnknownPrintFormat(val.to_string()))?;
            }
            "-show_format" => query.show_format = true,
            "-show_streams" => query.show_streams = true,
            "-show_packets" => query.show_packets = true,
            "-v" | "-loglevel" => {
                let level = it
                    .next()
                    .ok_or_else(|| FfprobeArgError::MissingValue(arg.to_string()))?;
                query.verbosity = Some(level.to_string());
            }
            "-select_streams" => {
                let spec = it
                    .next()
                    .ok_or_else(|| FfprobeArgError::MissingValue(arg.to_string()))?;
                query.select_streams = Some(spec.to_string());
            }
            // Unknown flags starting with '-' are silently ignored.
            s if s.starts_with('-') => {
                // Consume one value if the next token doesn't look like a flag.
                if it.peek().map_or(false, |n| !n.starts_with('-')) {
                    let _ = it.next();
                }
            }
            // Bare positional argument → input file (if not already set).
            path => {
                if query.input_path.is_none() {
                    query.input_path = Some(path.to_string());
                }
            }
        }
    }

    Ok(query)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_format_from_str() {
        assert_eq!(PrintFormat::from_str("json"), Some(PrintFormat::Json));
        assert_eq!(PrintFormat::from_str("csv"), Some(PrintFormat::Csv));
        assert_eq!(PrintFormat::from_str("flat"), Some(PrintFormat::Flat));
        assert_eq!(PrintFormat::from_str("JSON"), Some(PrintFormat::Json));
        assert!(PrintFormat::from_str("xml").is_none());
    }

    #[test]
    fn test_print_format_display() {
        assert_eq!(PrintFormat::Json.to_string(), "json");
        assert_eq!(PrintFormat::Csv.to_string(), "csv");
        assert_eq!(PrintFormat::Flat.to_string(), "flat");
    }

    #[test]
    fn test_probe_stream_video_fields() {
        let s = ProbeStream::new_video("h264", 1920, 1080, "16:9", 30.0);
        assert_eq!(s.codec_name, "h264");
        assert_eq!(s.width, Some(1920));
        assert_eq!(s.height, Some(1080));
        assert_eq!(s.display_aspect_ratio.as_deref(), Some("16:9"));
        assert!(matches!(s.codec_type, CodecType::Video));
    }

    #[test]
    fn test_probe_stream_audio_fields() {
        let s = ProbeStream::new_audio("aac", 48000, 2, "stereo");
        assert_eq!(s.codec_name, "aac");
        assert_eq!(s.sample_rate, Some(48000));
        assert_eq!(s.channels, Some(2));
        assert_eq!(s.channel_layout.as_deref(), Some("stereo"));
        assert!(matches!(s.codec_type, CodecType::Audio));
    }

    #[test]
    fn test_probe_stream_video_json_contains_key_fields() {
        let s = ProbeStream::new_video("av1", 3840, 2160, "16:9", 24.0);
        let json = s.to_json();
        assert!(json.contains("\"codec_name\""), "should have codec_name");
        assert!(json.contains("\"av1\""), "should have av1 value");
        assert!(json.contains("3840"), "should have width");
        assert!(json.contains("2160"), "should have height");
        assert!(json.contains("\"codec_type\""), "should have codec_type");
        assert!(json.contains("\"video\""), "should have video type");
    }

    #[test]
    fn test_probe_stream_audio_json_contains_key_fields() {
        let s = ProbeStream::new_audio("opus", 48000, 2, "stereo");
        let json = s.to_json();
        assert!(json.contains("\"codec_name\""));
        assert!(json.contains("\"opus\""));
        assert!(json.contains("\"audio\""));
        assert!(json.contains("48000"));
    }

    #[test]
    fn test_probe_format_json() {
        let f = ProbeFormat::new("test.mp4", "mp4", 10_000_000, 60.0);
        let json = f.to_json();
        assert!(json.contains("\"filename\""));
        assert!(json.contains("test.mp4"));
        assert!(json.contains("\"format_name\""));
        assert!(json.contains("mp4"));
        assert!(json.contains("\"duration\""));
    }

    #[test]
    fn test_probe_output_json_structure() {
        let v_stream = ProbeStream::new_video("av1", 1920, 1080, "16:9", 30.0);
        let a_stream = ProbeStream::new_audio("opus", 48000, 2, "stereo");
        let format = ProbeFormat::new("movie.mkv", "matroska", 500_000_000, 7200.0);

        let mut output = ProbeOutput::new();
        output.streams.push(v_stream);
        output.streams.push(a_stream);
        output.format = Some(format);

        let json = output.to_print_format(PrintFormat::Json);
        assert!(json.starts_with('{'), "should start with a brace");
        assert!(json.ends_with('}'), "should end with a brace");
        assert!(json.contains("\"streams\""), "should have streams key");
        assert!(json.contains("\"format\""), "should have format key");
        assert!(json.contains("\"codec_name\""), "codec_name present");
    }

    #[test]
    fn test_probe_output_csv_format() {
        let v_stream = ProbeStream::new_video("vp9", 1280, 720, "16:9", 25.0);
        let format = ProbeFormat::new("vid.webm", "webm", 1_000_000, 30.0);

        let mut output = ProbeOutput::new();
        output.streams.push(v_stream);
        output.format = Some(format);

        let csv = output.to_print_format(PrintFormat::Csv);
        assert!(
            csv.contains("stream,"),
            "CSV should start stream lines with 'stream,'"
        );
        assert!(csv.contains("format,"), "CSV should have a format line");
        assert!(csv.contains("vp9"), "should contain codec name");
    }

    #[test]
    fn test_probe_output_flat_format() {
        let a_stream = ProbeStream::new_audio("flac", 44100, 2, "stereo");

        let mut output = ProbeOutput::new();
        output.streams.push(a_stream);

        let flat = output.to_print_format(PrintFormat::Flat);
        assert!(
            flat.contains("codec_name=flac"),
            "flat should have codec_name=flac"
        );
        assert!(
            flat.contains("codec_type=audio"),
            "flat should have codec_type=audio"
        );
    }

    #[test]
    fn test_probe_output_empty() {
        let output = ProbeOutput::new();
        let json = output.to_print_format(PrintFormat::Json);
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
    }

    #[test]
    fn test_format_fps_ratio_ntsc() {
        assert_eq!(format_fps_ratio(29.97), "30000/1001");
        assert_eq!(format_fps_ratio(23.976), "24000/1001");
        assert_eq!(format_fps_ratio(59.94), "60000/1001");
    }

    #[test]
    fn test_format_fps_ratio_integer() {
        assert_eq!(format_fps_ratio(25.0), "25/1");
        assert_eq!(format_fps_ratio(30.0), "30/1");
        assert_eq!(format_fps_ratio(60.0), "60/1");
    }

    #[test]
    fn test_json_escape() {
        assert_eq!(json_escape("hello"), "hello");
        assert_eq!(json_escape("say \"hi\""), "say \\\"hi\\\"");
        assert_eq!(json_escape("a\\b"), "a\\\\b");
    }

    #[test]
    fn test_csv_escape_plain() {
        assert_eq!(csv_escape("hello"), "hello");
    }

    #[test]
    fn test_csv_escape_comma() {
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
    }

    #[test]
    fn test_codec_long_names() {
        let n = codec_long_name("av1");
        assert!(
            n.contains("AV1") || n.contains("Alliance"),
            "av1 long name: {}",
            n
        );

        let n2 = codec_long_name("opus");
        assert!(n2.to_lowercase().contains("opus"), "opus long name: {}", n2);
    }

    #[test]
    fn test_probe_stream_csv_format() {
        let s = ProbeStream::new_video("h264", 1920, 1080, "16:9", 30.0);
        let csv = s.to_csv();
        // Should be: stream,<idx>,<codec>,<type>,<width>,<height>,<sample_rate>
        let parts: Vec<&str> = csv.split(',').collect();
        assert_eq!(parts[0], "stream");
        assert_eq!(parts[2], "h264");
        assert_eq!(parts[3], "video");
        assert_eq!(parts[4], "1920");
        assert_eq!(parts[5], "1080");
    }

    #[test]
    fn test_probe_format_csv_fields() {
        let f = ProbeFormat::new("out.mkv", "matroska", 20_000_000, 120.0);
        let csv = f.to_csv();
        assert!(
            csv.starts_with("format,"),
            "format CSV should start with 'format,'"
        );
        assert!(csv.contains("matroska"), "should have format name");
    }

    #[test]
    fn test_probe_output_json_only_streams() {
        let mut output = ProbeOutput::new();
        output
            .streams
            .push(ProbeStream::new_video("vp9", 1280, 720, "16:9", 24.0));
        let json = output.to_json();
        assert!(json.contains("\"streams\""));
        assert!(!json.contains("\"format\""), "no format if not added");
    }

    #[test]
    fn test_probe_output_json_only_format() {
        let mut output = ProbeOutput::new();
        output.format = Some(ProbeFormat::new("audio.flac", "flac", 5_000_000, 200.0));
        let json = output.to_json();
        assert!(json.contains("\"format\""));
        assert!(!json.contains("\"streams\""), "no streams if none added");
    }

    #[test]
    fn test_codec_type_display() {
        assert_eq!(CodecType::Video.to_string(), "video");
        assert_eq!(CodecType::Audio.to_string(), "audio");
        assert_eq!(CodecType::Subtitle.to_string(), "subtitle");
    }

    // ── translate_ffprobe_args ───────────────────────────────────────────────

    #[test]
    fn test_translate_ffprobe_args_basic_json() {
        let args = &[
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-i",
            "movie.mkv",
        ];
        let q = translate_ffprobe_args(args).expect("should parse");
        assert_eq!(q.print_format, PrintFormat::Json);
        assert!(q.show_format);
        assert!(q.show_streams);
        assert!(!q.show_packets);
        assert_eq!(q.input_path.as_deref(), Some("movie.mkv"));
        assert_eq!(q.verbosity.as_deref(), Some("quiet"));
    }

    #[test]
    fn test_translate_ffprobe_args_of_alias() {
        let args = &["-of", "csv", "-show_streams", "input.mp4"];
        let q = translate_ffprobe_args(args).expect("should parse");
        assert_eq!(q.print_format, PrintFormat::Csv);
        assert!(q.show_streams);
        assert_eq!(q.input_path.as_deref(), Some("input.mp4"));
    }

    #[test]
    fn test_translate_ffprobe_args_show_packets() {
        let args = &["-show_packets", "-i", "stream.ts"];
        let q = translate_ffprobe_args(args).expect("should parse");
        assert!(q.show_packets);
        assert_eq!(q.input_path.as_deref(), Some("stream.ts"));
    }

    #[test]
    fn test_translate_ffprobe_args_select_streams() {
        let args = &["-select_streams", "v:0", "-show_streams", "-i", "vid.mkv"];
        let q = translate_ffprobe_args(args).expect("should parse");
        assert_eq!(q.select_streams.as_deref(), Some("v:0"));
        assert!(q.show_streams);
    }

    #[test]
    fn test_translate_ffprobe_args_flat_format() {
        let args = &["-print_format", "flat", "-show_format", "test.flac"];
        let q = translate_ffprobe_args(args).expect("should parse");
        assert_eq!(q.print_format, PrintFormat::Flat);
        assert!(q.show_format);
    }

    #[test]
    fn test_translate_ffprobe_args_positional_input() {
        // Input file as positional arg (common usage: ffprobe movie.mkv)
        let args = &["movie.mkv"];
        let q = translate_ffprobe_args(args).expect("should parse");
        assert_eq!(q.input_path.as_deref(), Some("movie.mkv"));
    }

    #[test]
    fn test_translate_ffprobe_args_unknown_print_format() {
        let args = &["-print_format", "xml_unsupported"];
        let err = translate_ffprobe_args(args).expect_err("should fail");
        assert!(matches!(err, FfprobeArgError::UnknownPrintFormat(_)));
    }

    #[test]
    fn test_translate_ffprobe_args_missing_value() {
        let args = &["-print_format"];
        let err = translate_ffprobe_args(args).expect_err("should fail on missing value");
        assert!(matches!(err, FfprobeArgError::MissingValue(_)));
    }

    #[test]
    fn test_translate_ffprobe_args_defaults() {
        let q = translate_ffprobe_args(&[]).expect("should parse empty");
        assert_eq!(q.print_format, PrintFormat::Json);
        assert!(!q.show_format);
        assert!(!q.show_streams);
        assert!(q.input_path.is_none());
    }
}
