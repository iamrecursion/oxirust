//! Build FFmpeg-compatible argument lists from OxiMedia configuration.
//!
//! [`FfmpegArgumentBuilder`] provides a fluent API for constructing an ordered
//! list of FFmpeg-style command-line arguments.  The resulting `Vec<String>`
//! can be passed directly to a subprocess, or rendered as a human-readable
//! `ffmpeg …` command string for logging/display.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_compat_ffmpeg::argument_builder::FfmpegArgumentBuilder;
//!
//! let mut b = FfmpegArgumentBuilder::new();
//! b.input("input.mkv")
//!  .video_codec("libaom-av1")
//!  .crf(30)
//!  .audio_codec("libopus")
//!  .audio_bitrate(128)
//!  .output("output.webm");
//!
//! let cmd = b.to_command_string();
//! assert!(cmd.starts_with("ffmpeg "));
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// Builder type
// ─────────────────────────────────────────────────────────────────────────────

/// Fluent builder that accumulates FFmpeg-style command-line arguments.
///
/// Arguments are stored in three ordered buckets:
///
/// * **`global_args`** — placed immediately after `ffmpeg`, before any `-i`.
/// * **`input_args`** — the `-i path` pair(s).
/// * **`output_args`** — all encoding options and the output path.
///
/// `to_args` concatenates them in this order: global → input → output.
#[derive(Debug, Clone, Default)]
pub struct FfmpegArgumentBuilder {
    /// Global flags such as `-y` (overwrite), `-loglevel`, etc.
    pub global_args: Vec<String>,
    /// Input file arguments: `-i <path>` pairs.
    pub input_args: Vec<String>,
    /// Output encoding options and the output file path.
    pub output_args: Vec<String>,
}

impl FfmpegArgumentBuilder {
    /// Create a new, empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    // ── Input / output ───────────────────────────────────────────────────────

    /// Append `-i <path>` to the input argument list.
    ///
    /// Multiple calls are allowed for multi-input graphs.
    pub fn input(&mut self, path: &str) -> &mut Self {
        self.input_args.push("-i".to_string());
        self.input_args.push(path.to_string());
        self
    }

    /// Append `<path>` to the output argument list (must be called last).
    ///
    /// Calling this method multiple times appends multiple output paths, which
    /// is valid FFmpeg syntax for fan-out transcodes.
    pub fn output(&mut self, path: &str) -> &mut Self {
        self.output_args.push(path.to_string());
        self
    }

    // ── Codec flags ──────────────────────────────────────────────────────────

    /// Append `-c:v <codec>` to the output arguments.
    pub fn video_codec(&mut self, codec: &str) -> &mut Self {
        self.output_args.push("-c:v".to_string());
        self.output_args.push(codec.to_string());
        self
    }

    /// Append `-c:a <codec>` to the output arguments.
    pub fn audio_codec(&mut self, codec: &str) -> &mut Self {
        self.output_args.push("-c:a".to_string());
        self.output_args.push(codec.to_string());
        self
    }

    // ── Bitrate flags ─────────────────────────────────────────────────────────

    /// Append `-b:v <kbps>k` to the output arguments.
    pub fn video_bitrate(&mut self, kbps: u32) -> &mut Self {
        self.output_args.push("-b:v".to_string());
        self.output_args.push(format!("{}k", kbps));
        self
    }

    /// Append `-b:a <kbps>k` to the output arguments.
    pub fn audio_bitrate(&mut self, kbps: u32) -> &mut Self {
        self.output_args.push("-b:a".to_string());
        self.output_args.push(format!("{}k", kbps));
        self
    }

    // ── Quality flags ─────────────────────────────────────────────────────────

    /// Append `-crf <crf>` to the output arguments.
    ///
    /// The CRF scale is codec-dependent; for AV1 a value of 23–35 is typical.
    pub fn crf(&mut self, crf: u8) -> &mut Self {
        self.output_args.push("-crf".to_string());
        self.output_args.push(crf.to_string());
        self
    }

    // ── Video geometry ────────────────────────────────────────────────────────

    /// Append `-vf scale=<w>:<h>` to the output arguments.
    ///
    /// Pass `0` for either dimension to instruct FFmpeg to preserve aspect
    /// ratio (i.e. `scale=1280:0`).
    pub fn resolution(&mut self, w: u32, h: u32) -> &mut Self {
        self.output_args.push("-vf".to_string());
        self.output_args.push(format!("scale={}:{}", w, h));
        self
    }

    // ── Frame rate ────────────────────────────────────────────────────────────

    /// Append `-r <fps>` to the output arguments.
    ///
    /// The value is formatted with up to three decimal places and trailing
    /// zeros stripped (e.g. `29.97`, not `29.970000`).
    pub fn fps(&mut self, fps: f32) -> &mut Self {
        self.output_args.push("-r".to_string());
        // Format with enough precision for NTSC-style rates (e.g. 29.97),
        // then strip trailing zeros after the decimal point.
        let formatted = format_fps(fps);
        self.output_args.push(formatted);
        self
    }

    // ── Audio sample rate ─────────────────────────────────────────────────────

    /// Append `-ar <hz>` to the output arguments.
    pub fn audio_sample_rate(&mut self, hz: u32) -> &mut Self {
        self.output_args.push("-ar".to_string());
        self.output_args.push(hz.to_string());
        self
    }

    // ── Global flags ──────────────────────────────────────────────────────────

    /// Append `-y` (overwrite output without prompting) to the global args.
    pub fn overwrite(&mut self) -> &mut Self {
        self.global_args.push("-y".to_string());
        self
    }

    /// Append `-loglevel <level>` to the global args.
    pub fn loglevel(&mut self, level: &str) -> &mut Self {
        self.global_args.push("-loglevel".to_string());
        self.global_args.push(level.to_string());
        self
    }

    /// Append an arbitrary video filter string via `-vf <expr>`.
    ///
    /// This supplements the high-level `resolution` helper for cases where
    /// the caller needs a full filter chain (e.g. `"scale=1920:1080,fps=30"`).
    pub fn video_filter(&mut self, expr: &str) -> &mut Self {
        self.output_args.push("-vf".to_string());
        self.output_args.push(expr.to_string());
        self
    }

    /// Append an arbitrary audio filter string via `-af <expr>`.
    pub fn audio_filter(&mut self, expr: &str) -> &mut Self {
        self.output_args.push("-af".to_string());
        self.output_args.push(expr.to_string());
        self
    }

    /// Append a `-filter_complex <expr>` global filter graph.
    pub fn filter_complex(&mut self, expr: &str) -> &mut Self {
        self.output_args.push("-filter_complex".to_string());
        self.output_args.push(expr.to_string());
        self
    }

    /// Append `-t <seconds>` (maximum output duration) to the output args.
    pub fn duration_secs(&mut self, secs: f64) -> &mut Self {
        self.output_args.push("-t".to_string());
        self.output_args.push(format!("{:.3}", secs));
        self
    }

    /// Append `-ss <position>` (seek position) to the output args.
    pub fn seek(&mut self, position: &str) -> &mut Self {
        self.output_args.push("-ss".to_string());
        self.output_args.push(position.to_string());
        self
    }

    /// Append `-f <format>` (force container format) to the output args.
    pub fn format(&mut self, fmt: &str) -> &mut Self {
        self.output_args.push("-f".to_string());
        self.output_args.push(fmt.to_string());
        self
    }

    /// Append `-metadata key=value` to the output args.
    pub fn metadata(&mut self, key: &str, value: &str) -> &mut Self {
        self.output_args.push("-metadata".to_string());
        self.output_args.push(format!("{}={}", key, value));
        self
    }

    // ── Build ─────────────────────────────────────────────────────────────────

    /// Assemble the complete argument list in canonical FFmpeg order:
    ///
    /// `[global_args] [input_args] [output_args]`
    ///
    /// The returned `Vec` does **not** include the `"ffmpeg"` binary name.
    pub fn to_args(&self) -> Vec<String> {
        let mut args = Vec::with_capacity(
            self.global_args.len() + self.input_args.len() + self.output_args.len(),
        );
        args.extend_from_slice(&self.global_args);
        args.extend_from_slice(&self.input_args);
        args.extend_from_slice(&self.output_args);
        args
    }

    /// Build a human-readable `ffmpeg …` command string.
    ///
    /// Arguments that contain spaces are enclosed in double-quotes.  This is
    /// suitable for logging and display — for actual subprocess execution use
    /// `to_args` instead to avoid shell-escaping issues.
    pub fn to_command_string(&self) -> String {
        let args = self.to_args();
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

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Format an fps value with minimal precision (strip trailing zeros).
fn format_fps(fps: f32) -> String {
    // Use three decimal places then strip trailing zeros + optional dot.
    let s = format!("{:.3}", fps);
    let s = s.trim_end_matches('0');
    let s = s.trim_end_matches('.');
    s.to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_basic() -> FfmpegArgumentBuilder {
        let mut b = FfmpegArgumentBuilder::new();
        b.input("input.mkv").output("output.webm");
        b
    }

    // ── basic argument construction ───────────────────────────────────────────

    #[test]
    fn test_input_output_args() {
        let b = build_basic();
        let args = b.to_args();
        assert_eq!(args[0], "-i");
        assert_eq!(args[1], "input.mkv");
        assert_eq!(args[2], "output.webm");
    }

    #[test]
    fn test_video_codec_flag() {
        let mut b = FfmpegArgumentBuilder::new();
        b.input("in.mkv")
            .video_codec("libaom-av1")
            .output("out.webm");
        let args = b.to_args();
        let idx = args
            .iter()
            .position(|a| a == "-c:v")
            .expect("-c:v must be present");
        assert_eq!(args[idx + 1], "libaom-av1");
    }

    #[test]
    fn test_audio_codec_flag() {
        let mut b = FfmpegArgumentBuilder::new();
        b.input("in.mkv").audio_codec("libopus").output("out.webm");
        let args = b.to_args();
        let idx = args
            .iter()
            .position(|a| a == "-c:a")
            .expect("-c:a must be present");
        assert_eq!(args[idx + 1], "libopus");
    }

    #[test]
    fn test_video_bitrate_format() {
        let mut b = FfmpegArgumentBuilder::new();
        b.input("in.mkv").video_bitrate(2000).output("out.webm");
        let args = b.to_args();
        let idx = args
            .iter()
            .position(|a| a == "-b:v")
            .expect("-b:v must be present");
        assert_eq!(args[idx + 1], "2000k");
    }

    #[test]
    fn test_audio_bitrate_format() {
        let mut b = FfmpegArgumentBuilder::new();
        b.input("in.mkv").audio_bitrate(128).output("out.webm");
        let args = b.to_args();
        let idx = args
            .iter()
            .position(|a| a == "-b:a")
            .expect("-b:a must be present");
        assert_eq!(args[idx + 1], "128k");
    }

    #[test]
    fn test_crf_flag() {
        let mut b = FfmpegArgumentBuilder::new();
        b.input("in.mkv").crf(28).output("out.webm");
        let args = b.to_args();
        let idx = args
            .iter()
            .position(|a| a == "-crf")
            .expect("-crf must be present");
        assert_eq!(args[idx + 1], "28");
    }

    #[test]
    fn test_resolution_generates_vf_scale() {
        let mut b = FfmpegArgumentBuilder::new();
        b.input("in.mkv").resolution(1920, 1080).output("out.webm");
        let args = b.to_args();
        let idx = args
            .iter()
            .position(|a| a == "-vf")
            .expect("-vf must be present");
        assert_eq!(args[idx + 1], "scale=1920:1080");
    }

    #[test]
    fn test_fps_integer() {
        let mut b = FfmpegArgumentBuilder::new();
        b.input("in.mkv").fps(30.0).output("out.webm");
        let args = b.to_args();
        let idx = args
            .iter()
            .position(|a| a == "-r")
            .expect("-r must be present");
        assert_eq!(args[idx + 1], "30");
    }

    #[test]
    fn test_fps_fractional_ntsc() {
        let mut b = FfmpegArgumentBuilder::new();
        b.input("in.mkv").fps(29.97).output("out.webm");
        let args = b.to_args();
        let idx = args
            .iter()
            .position(|a| a == "-r")
            .expect("-r must be present");
        // Should be "29.97" not "29.970" or "29.970000".
        assert_eq!(args[idx + 1], "29.97");
    }

    #[test]
    fn test_audio_sample_rate() {
        let mut b = FfmpegArgumentBuilder::new();
        b.input("in.mkv")
            .audio_sample_rate(48000)
            .output("out.webm");
        let args = b.to_args();
        let idx = args
            .iter()
            .position(|a| a == "-ar")
            .expect("-ar must be present");
        assert_eq!(args[idx + 1], "48000");
    }

    // ── to_command_string ─────────────────────────────────────────────────────

    #[test]
    fn test_command_string_starts_with_ffmpeg() {
        let b = build_basic();
        assert!(b.to_command_string().starts_with("ffmpeg "));
    }

    #[test]
    fn test_command_string_contains_input_and_output() {
        let b = build_basic();
        let cmd = b.to_command_string();
        assert!(cmd.contains("-i input.mkv"), "cmd: {}", cmd);
        assert!(cmd.contains("output.webm"), "cmd: {}", cmd);
    }

    // ── global args ordering ──────────────────────────────────────────────────

    #[test]
    fn test_global_args_precede_input() {
        let mut b = FfmpegArgumentBuilder::new();
        b.overwrite().input("in.mkv").output("out.webm");
        let args = b.to_args();
        let y_idx = args.iter().position(|a| a == "-y").expect("-y");
        let i_idx = args.iter().position(|a| a == "-i").expect("-i");
        assert!(y_idx < i_idx, "-y should come before -i");
    }

    // ── fluent chaining ───────────────────────────────────────────────────────

    #[test]
    fn test_full_transcode_command() {
        let mut b = FfmpegArgumentBuilder::new();
        b.overwrite()
            .input("input.mkv")
            .video_codec("libaom-av1")
            .crf(30)
            .video_bitrate(0) // let CRF control quality
            .audio_codec("libopus")
            .audio_bitrate(128)
            .audio_sample_rate(48000)
            .resolution(1920, 1080)
            .fps(24.0)
            .output("output.webm");

        let args = b.to_args();
        // Check all expected flags are present.
        assert!(args.contains(&"-y".to_string()));
        assert!(args.contains(&"-c:v".to_string()));
        assert!(args.contains(&"-crf".to_string()));
        assert!(args.contains(&"-c:a".to_string()));
        assert!(args.contains(&"-b:a".to_string()));
        assert!(args.contains(&"-ar".to_string()));
        assert!(args.contains(&"-vf".to_string()));
        assert!(args.contains(&"-r".to_string()));
        assert!(args.last() == Some(&"output.webm".to_string()));
    }
}
