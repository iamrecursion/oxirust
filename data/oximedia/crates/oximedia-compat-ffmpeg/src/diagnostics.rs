//! Structured diagnostics for FFmpeg-compat translation.
//!
//! During argument parsing and translation, various warnings and errors may be
//! generated — for example, patent-encumbered codecs, unsupported options, or
//! unknown filter names. [`Diagnostic`] collects these in a structured way so
//! callers can present them to users or programmatically inspect them.

use thiserror::Error;

/// The category/kind of a diagnostic message, with rich semantic detail.
#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticKind {
    /// A patent-encumbered codec was requested and substituted with a free alternative.
    PatentCodecSubstituted {
        /// The originally requested FFmpeg codec name.
        from: String,
        /// The OxiMedia patent-free codec used instead.
        to: String,
    },
    /// An FFmpeg option is not supported and was silently ignored.
    UnknownOptionIgnored {
        /// The option string that was ignored.
        option: String,
    },
    /// A filter in the filtergraph is not supported by OxiMedia.
    FilterNotSupported {
        /// The filter name that was skipped.
        filter: String,
    },
    /// A feature is known but not yet implemented.
    UnsupportedFeature {
        /// Human-readable description of what is not supported.
        description: String,
    },
    /// An informational note about how an option was mapped.
    Info {
        /// The informational message.
        message: String,
    },
    /// A hard error that prevents translation from completing.
    Error {
        /// Description of the error.
        message: String,
    },
    /// A non-fatal warning.
    Warning {
        /// Description of the warning.
        message: String,
    },
}

impl DiagnosticKind {
    /// Return `true` if this kind represents a fatal error.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Short severity label for display.
    pub fn severity_label(&self) -> &'static str {
        match self {
            Self::Error { .. } => "error",
            Self::Warning { .. } => "warning",
            Self::PatentCodecSubstituted { .. } => "warning",
            Self::UnknownOptionIgnored { .. } => "warning",
            Self::FilterNotSupported { .. } => "warning",
            Self::UnsupportedFeature { .. } => "warning",
            Self::Info { .. } => "info",
        }
    }
}

impl std::fmt::Display for DiagnosticKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.severity_label())
    }
}

/// A single diagnostic message produced during FFmpeg-compat translation.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Semantic kind of this diagnostic (determines severity and message format).
    pub kind: DiagnosticKind,
    /// Optional additional hint about how to resolve the issue.
    pub suggestion: Option<String>,
}

impl Diagnostic {
    /// Create a `PatentCodecSubstituted` diagnostic.
    pub fn patent_substituted(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::PatentCodecSubstituted {
                from: from.into(),
                to: to.into(),
            },
            suggestion: None,
        }
    }

    /// Create an `UnknownOptionIgnored` diagnostic.
    pub fn unknown_option(option: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::UnknownOptionIgnored {
                option: option.into(),
            },
            suggestion: None,
        }
    }

    /// Create a `FilterNotSupported` diagnostic.
    pub fn filter_not_supported(filter: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::FilterNotSupported {
                filter: filter.into(),
            },
            suggestion: None,
        }
    }

    /// Create an `UnsupportedFeature` diagnostic.
    pub fn unsupported_feature(description: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::UnsupportedFeature {
                description: description.into(),
            },
            suggestion: None,
        }
    }

    /// Create an info diagnostic.
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::Info {
                message: message.into(),
            },
            suggestion: None,
        }
    }

    /// Create an error-level diagnostic.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::Error {
                message: message.into(),
            },
            suggestion: None,
        }
    }

    /// Create a warning-level diagnostic.
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::Warning {
                message: message.into(),
            },
            suggestion: None,
        }
    }

    /// Attach a suggestion to this diagnostic.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Return `true` if this diagnostic represents a fatal error.
    pub fn is_error(&self) -> bool {
        self.kind.is_error()
    }

    /// Format this diagnostic in FFmpeg's stderr style.
    ///
    /// ```text
    /// oximedia-ff: Codec 'libx264' is a patent codec. Using 'av1' instead.
    /// oximedia-ff: Option '-hwaccel' not supported. Ignoring.
    /// ```
    pub fn format_ffmpeg_style(&self, program: &str) -> String {
        let base = match &self.kind {
            DiagnosticKind::PatentCodecSubstituted { from, to } => {
                format!(
                    "{}: Codec '{}' is a patent codec. Using '{}' instead.",
                    program, from, to
                )
            }
            DiagnosticKind::UnknownOptionIgnored { option } => {
                format!("{}: Option '{}' not supported. Ignoring.", program, option)
            }
            DiagnosticKind::FilterNotSupported { filter } => {
                format!("{}: Filter '{}' not supported. Skipping.", program, filter)
            }
            DiagnosticKind::UnsupportedFeature { description } => {
                format!("{}: {}.", program, description)
            }
            DiagnosticKind::Info { message } => {
                format!("{}: {}", program, message)
            }
            DiagnosticKind::Error { message } => {
                format!("{}: error: {}", program, message)
            }
            DiagnosticKind::Warning { message } => {
                format!("{}: warning: {}", program, message)
            }
        };

        if let Some(hint) = &self.suggestion {
            format!("{}\n  hint: {}", base, hint)
        } else {
            base
        }
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_ffmpeg_style("oximedia-ff"))
    }
}

/// Error type for fatal translation failures.
#[derive(Debug, Error)]
pub enum TranslationError {
    /// The input arguments could not be parsed.
    #[error("argument parse error: {0}")]
    ParseError(String),

    /// A required input file was not specified.
    #[error("no input file specified")]
    NoInput,

    /// A required output file was not specified.
    #[error("no output file specified")]
    NoOutput,

    /// A filter expression could not be parsed.
    #[error("filter parse error: {0}")]
    FilterParseError(String),

    /// An anyhow error converted into a translation error.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// ─────────────────────────────────────────────────────────────────────────────
// Fuzzy "did you mean?" codec suggestion engine
// ─────────────────────────────────────────────────────────────────────────────

/// Well-known codec names used as candidates for "did you mean?" suggestions.
static KNOWN_CODEC_NAMES: &[&str] = &[
    "libaom-av1",
    "libsvtav1",
    "librav1e",
    "av1",
    "av1_nvenc",
    "av1_vaapi",
    "av1_amf",
    "libvpx-vp9",
    "vp9",
    "libvpx",
    "vp8",
    "libx264",
    "h264",
    "h264_nvenc",
    "h264_vaapi",
    "h264_qsv",
    "libx265",
    "hevc",
    "hevc_nvenc",
    "hevc_vaapi",
    "hevc_qsv",
    "prores",
    "dnxhd",
    "dnxhr",
    "ffv1",
    "huffyuv",
    "libopus",
    "opus",
    "aac",
    "libfdk_aac",
    "mp3",
    "libmp3lame",
    "flac",
    "alac",
    "vorbis",
    "libvorbis",
    "pcm_s16le",
    "pcm_s24le",
    "pcm_s32le",
    "pcm_f32le",
    "ac3",
    "eac3",
    "dts",
    "truehd",
    "copy",
];

/// Compute the Levenshtein edit distance between two strings (case-insensitive).
fn edit_distance(a: &str, b: &str) -> usize {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let a_chars: Vec<char> = a_lower.chars().collect();
    let b_chars: Vec<char> = b_lower.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Find the best "did you mean?" suggestion for a mistyped codec name.
///
/// Returns `Some(suggestion)` if a known codec name is within edit distance 3,
/// or if the input is a substring of (or contains) a known name.
/// Returns `None` if no close match is found.
pub fn suggest_codec(mistyped: &str) -> Option<&'static str> {
    let mistyped_lower = mistyped.to_lowercase();

    // First pass: exact substring containment (e.g., "x264" -> "libx264")
    for &candidate in KNOWN_CODEC_NAMES {
        let candidate_lower = candidate.to_lowercase();
        if candidate_lower.contains(&mistyped_lower) || mistyped_lower.contains(&candidate_lower) {
            return Some(candidate);
        }
    }

    // Second pass: edit distance with threshold
    let max_distance = if mistyped.len() <= 3 { 1 } else { 3 };
    let mut best_dist = usize::MAX;
    let mut best_match: Option<&'static str> = None;

    for &candidate in KNOWN_CODEC_NAMES {
        let dist = edit_distance(mistyped, candidate);
        if dist < best_dist && dist <= max_distance {
            best_dist = dist;
            best_match = Some(candidate);
        }
    }

    best_match
}

/// Create a diagnostic for an unknown codec with an automatic "did you mean?" suggestion.
pub fn unknown_codec_diagnostic(codec_name: &str) -> Diagnostic {
    let mut diag = Diagnostic {
        kind: DiagnosticKind::UnknownOptionIgnored {
            option: codec_name.to_string(),
        },
        suggestion: Some("Use a patent-free codec: av1, vp9, vp8, opus, vorbis, flac".to_string()),
    };

    if let Some(suggestion) = suggest_codec(codec_name) {
        diag.suggestion = Some(format!(
            "Did you mean '{}'? Patent-free alternatives: av1, vp9, opus, vorbis, flac",
            suggestion
        ));
    }

    diag
}

/// Collect diagnostics produced during a translation pass.
#[derive(Debug, Default)]
pub struct DiagnosticSink {
    items: Vec<Diagnostic>,
}

impl DiagnosticSink {
    /// Create an empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a diagnostic into the sink.
    pub fn push(&mut self, diag: Diagnostic) {
        self.items.push(diag);
    }

    /// Return all diagnostics.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.items
    }

    /// Return `true` if any error-level diagnostics were collected.
    pub fn has_errors(&self) -> bool {
        self.items.iter().any(|d| d.is_error())
    }

    /// Consume the sink and return all diagnostics.
    pub fn into_diagnostics(self) -> Vec<Diagnostic> {
        self.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patent_format_message() {
        let d = Diagnostic::patent_substituted("libx264", "av1");
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(msg.contains("libx264"), "should mention original codec");
        assert!(msg.contains("av1"), "should mention replacement");
        assert!(msg.contains("oximedia-ff"), "should mention program name");
        // Patent substitution is a warning, not an error
        assert!(!d.is_error(), "patent substitution is a warning, not error");
    }

    #[test]
    fn test_patent_substituted_kind() {
        let d = Diagnostic::patent_substituted("aac", "opus");
        match &d.kind {
            DiagnosticKind::PatentCodecSubstituted { from, to } => {
                assert_eq!(from, "aac");
                assert_eq!(to, "opus");
            }
            _ => panic!("expected PatentCodecSubstituted"),
        }
    }

    #[test]
    fn test_unknown_option_format() {
        let d = Diagnostic::unknown_option("-movflags");
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(msg.contains("-movflags"), "should mention the option");
        assert!(
            msg.contains("Ignoring") || msg.contains("not supported"),
            "should indicate the option is ignored or not supported"
        );
        assert!(!d.is_error(), "unknown option is a warning, not error");
    }

    #[test]
    fn test_filter_not_supported_format() {
        let d = Diagnostic::filter_not_supported("drawtext");
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(msg.contains("drawtext"), "should mention the filter name");
        assert!(
            !d.is_error(),
            "filter not supported is a warning, not error"
        );
    }

    #[test]
    fn test_unsupported_feature_format() {
        let d = Diagnostic::unsupported_feature("Hardware decoding via NVDEC");
        let msg = d.format_ffmpeg_style("myapp");
        assert!(msg.contains("Hardware decoding via NVDEC"));
        assert!(msg.starts_with("myapp:"), "should start with program name");
    }

    #[test]
    fn test_error_diagnostic_is_error() {
        let d = Diagnostic::error("something went wrong");
        assert!(
            d.is_error(),
            "error diagnostic should report is_error() = true"
        );
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(
            msg.contains("error"),
            "error message should contain 'error'"
        );
        assert!(msg.contains("something went wrong"));
    }

    #[test]
    fn test_warning_diagnostic_not_error() {
        let d = Diagnostic::warning("this is a warning");
        assert!(!d.is_error(), "warning should not be an error");
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(msg.contains("warning"));
    }

    #[test]
    fn test_info_diagnostic_not_error() {
        let d = Diagnostic::info("informational message");
        assert!(!d.is_error());
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(msg.contains("informational message"));
    }

    #[test]
    fn test_diagnostic_with_suggestion() {
        let d = Diagnostic::unknown_option("-hwaccel")
            .with_suggestion("Use -c:v av1 for software AV1 encoding");
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(msg.contains("hint:"), "suggestion should appear as a hint");
        assert!(msg.contains("Use -c:v av1"), "hint text should be present");
    }

    #[test]
    fn test_diagnostic_sink_collects_items() {
        let mut sink = DiagnosticSink::new();
        sink.push(Diagnostic::warning("w1"));
        sink.push(Diagnostic::info("i1"));
        sink.push(Diagnostic::warning("w2"));
        assert_eq!(sink.diagnostics().len(), 3);
        assert!(!sink.has_errors());
    }

    #[test]
    fn test_diagnostic_sink_detects_errors() {
        let mut sink = DiagnosticSink::new();
        sink.push(Diagnostic::warning("harmless"));
        assert!(!sink.has_errors());
        sink.push(Diagnostic::error("fatal"));
        assert!(sink.has_errors());
    }

    #[test]
    fn test_diagnostic_sink_into_diagnostics() {
        let mut sink = DiagnosticSink::new();
        sink.push(Diagnostic::info("one"));
        sink.push(Diagnostic::info("two"));
        let items = sink.into_diagnostics();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_severity_labels() {
        assert_eq!(
            DiagnosticKind::Error { message: "".into() }.severity_label(),
            "error"
        );
        assert_eq!(
            DiagnosticKind::Warning { message: "".into() }.severity_label(),
            "warning"
        );
        assert_eq!(
            DiagnosticKind::Info { message: "".into() }.severity_label(),
            "info"
        );
        assert_eq!(
            DiagnosticKind::PatentCodecSubstituted {
                from: "".into(),
                to: "".into()
            }
            .severity_label(),
            "warning"
        );
        assert_eq!(
            DiagnosticKind::UnknownOptionIgnored { option: "".into() }.severity_label(),
            "warning"
        );
        assert_eq!(
            DiagnosticKind::FilterNotSupported { filter: "".into() }.severity_label(),
            "warning"
        );
    }

    #[test]
    fn test_format_starts_with_program_name() {
        let d = Diagnostic::patent_substituted("libx264", "av1");
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(
            msg.starts_with("oximedia-ff:"),
            "should start with program name colon"
        );
    }

    #[test]
    fn test_display_impl_uses_default_program() {
        let d = Diagnostic::warning("test warning");
        let display = format!("{}", d);
        // Display impl calls format_ffmpeg_style("oximedia-ff")
        assert!(display.contains("oximedia-ff"));
        assert!(display.contains("test warning"));
    }

    #[test]
    fn test_translation_error_display() {
        let e = TranslationError::NoInput;
        assert!(e.to_string().contains("no input file"));

        let e2 = TranslationError::NoOutput;
        assert!(e2.to_string().contains("no output file"));

        let e3 = TranslationError::ParseError("bad arg".into());
        assert!(e3.to_string().contains("bad arg"));

        let e4 = TranslationError::FilterParseError("bad filter".into());
        assert!(e4.to_string().contains("bad filter"));
    }

    // ── "did you mean?" suggestion engine tests ─────────────────────────────

    #[test]
    fn test_suggest_codec_exact_substring() {
        // "x264" is a substring of "libx264"
        let s = suggest_codec("x264");
        assert!(s.is_some(), "x264 should match a known codec");
        assert_eq!(s, Some("libx264"));
    }

    #[test]
    fn test_suggest_codec_close_typo() {
        // "libaom_av1" vs "libaom-av1" (close edit distance)
        let s = suggest_codec("libaom_av1");
        assert!(s.is_some(), "libaom_av1 should match");
    }

    #[test]
    fn test_suggest_codec_opus_typo() {
        let s = suggest_codec("opsu");
        assert!(s.is_some(), "opsu should suggest opus");
        assert_eq!(s, Some("opus"));
    }

    #[test]
    fn test_suggest_codec_no_match() {
        let s = suggest_codec("zzzzzzzzzzzzz");
        assert!(s.is_none(), "gibberish should not match");
    }

    #[test]
    fn test_suggest_codec_partial_hevc() {
        let s = suggest_codec("hev");
        assert!(s.is_some(), "hev should suggest hevc");
        // Should match something containing "hev"
        let suggestion = s.expect("already checked");
        assert!(
            suggestion.contains("hevc") || suggestion.contains("hev"),
            "suggestion should be related to hevc, got {}",
            suggestion
        );
    }

    #[test]
    fn test_suggest_codec_vp9_typo() {
        let s = suggest_codec("vp0");
        // Close edit distance to vp8 or vp9
        assert!(s.is_some(), "vp0 should suggest vp8 or vp9");
    }

    #[test]
    fn test_unknown_codec_diagnostic_with_suggestion() {
        let d = unknown_codec_diagnostic("libx26");
        assert!(d.suggestion.is_some());
        let hint = d.suggestion.as_deref().unwrap_or("");
        assert!(
            hint.contains("Did you mean"),
            "should contain 'Did you mean', got: {}",
            hint
        );
    }

    #[test]
    fn test_unknown_codec_diagnostic_no_suggestion() {
        let d = unknown_codec_diagnostic("completely_made_up_zzzzz");
        // Should still have a suggestion, but the generic one
        assert!(d.suggestion.is_some());
        let hint = d.suggestion.as_deref().unwrap_or("");
        assert!(
            hint.contains("patent-free") || hint.contains("Patent-free"),
            "should contain patent-free suggestion, got: {}",
            hint
        );
    }

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(edit_distance("opus", "opus"), 0);
    }

    #[test]
    fn test_edit_distance_one_char() {
        assert_eq!(edit_distance("opus", "opsu"), 2);
    }

    #[test]
    fn test_edit_distance_different() {
        assert!(edit_distance("av1", "zzz") > 0);
    }

    #[test]
    fn test_edit_distance_empty() {
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("", ""), 0);
    }

    #[test]
    fn test_diagnostic_kind_is_error_only_for_error_variant() {
        assert!(DiagnosticKind::Error { message: "".into() }.is_error());
        assert!(!DiagnosticKind::Warning { message: "".into() }.is_error());
        assert!(!DiagnosticKind::Info { message: "".into() }.is_error());
        assert!(!DiagnosticKind::PatentCodecSubstituted {
            from: "".into(),
            to: "".into()
        }
        .is_error());
        assert!(!DiagnosticKind::UnknownOptionIgnored { option: "".into() }.is_error());
        assert!(!DiagnosticKind::FilterNotSupported { filter: "".into() }.is_error());
        assert!(!DiagnosticKind::UnsupportedFeature {
            description: "".into()
        }
        .is_error());
    }
}
