//! Subtitle export to multiple text-based formats.
//!
//! [`SubtitleExporter`] converts an in-memory subtitle cue into the textual
//! representation required by a particular target format.  It is intentionally
//! stateless so the same instance can be reused across many cues.

#![allow(dead_code)]

/// The output format to target during export.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExportTarget {
    /// SubRip Text (`.srt`).
    Srt,
    /// Web Video Text Tracks (`.vtt`).
    Vtt,
    /// Advanced SubStation Alpha (`.ass`).
    Ass,
    /// Timed Text Markup Language (`.ttml` / `.xml`).
    Ttml,
}

impl ExportTarget {
    /// Return the conventional file extension (without the leading dot).
    #[must_use]
    pub fn extension(&self) -> &str {
        match self {
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Ass => "ass",
            Self::Ttml => "ttml",
        }
    }

    /// Return the MIME type associated with this format.
    #[must_use]
    pub fn mime_type(&self) -> &str {
        match self {
            Self::Srt => "text/plain",
            Self::Vtt => "text/vtt",
            Self::Ass => "text/x-ssa",
            Self::Ttml => "application/ttml+xml",
        }
    }
}

/// A single subtitle entry supplied to the exporter.
#[derive(Debug, Clone)]
pub struct ExportEntry {
    /// Sequential (one-based) index for formats that require it.
    pub sequence: usize,
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Plain-text subtitle content.
    pub text: String,
}

impl ExportEntry {
    /// Create a new `ExportEntry`.
    #[must_use]
    pub fn new(sequence: usize, start_ms: i64, end_ms: i64, text: impl Into<String>) -> Self {
        Self {
            sequence,
            start_ms,
            end_ms,
            text: text.into(),
        }
    }
}

/// Formats an `i64` millisecond value as `HH:MM:SS,mmm` (SRT style).
fn ms_to_srt_ts(ms: i64) -> String {
    let sign = if ms < 0 { "-" } else { "" };
    let ms = ms.unsigned_abs();
    let millis = ms % 1000;
    let secs = (ms / 1000) % 60;
    let mins = (ms / 60_000) % 60;
    let hours = ms / 3_600_000;
    format!("{sign}{hours:02}:{mins:02}:{secs:02},{millis:03}")
}

/// Formats an `i64` millisecond value as `HH:MM:SS.mmm` (VTT / TTML style).
fn ms_to_vtt_ts(ms: i64) -> String {
    let sign = if ms < 0 { "-" } else { "" };
    let ms = ms.unsigned_abs();
    let millis = ms % 1000;
    let secs = (ms / 1000) % 60;
    let mins = (ms / 60_000) % 60;
    let hours = ms / 3_600_000;
    format!("{sign}{hours:02}:{mins:02}:{secs:02}.{millis:03}")
}

/// Formats an `i64` millisecond value as `H:MM:SS.cc` (ASS style, centiseconds).
fn ms_to_ass_ts(ms: i64) -> String {
    let ms = ms.max(0) as u64;
    let cs = (ms / 10) % 100;
    let secs = (ms / 1000) % 60;
    let mins = (ms / 60_000) % 60;
    let hours = ms / 3_600_000;
    format!("{hours}:{mins:02}:{secs:02}.{cs:02}")
}

/// Stateless subtitle exporter that converts cue entries to format strings.
///
/// # Example
///
/// ```
/// use oximedia_subtitle::subtitle_export::{ExportEntry, ExportTarget, SubtitleExporter};
///
/// let exporter = SubtitleExporter::new();
/// let entry = ExportEntry::new(1, 1000, 4000, "Hello, world!");
/// let srt_block = exporter.format_entry(&entry, &ExportTarget::Srt);
/// assert!(srt_block.contains("00:00:01,000 --> 00:00:04,000"));
/// assert!(srt_block.contains("Hello, world!"));
/// ```
#[derive(Debug, Default)]
pub struct SubtitleExporter;

impl SubtitleExporter {
    /// Create a new `SubtitleExporter`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Render a single cue entry in the chosen target format.
    ///
    /// The returned string is a self-contained block ready to be concatenated
    /// with other blocks to form the complete subtitle file.
    #[must_use]
    pub fn format_entry(&self, entry: &ExportEntry, target: &ExportTarget) -> String {
        match target {
            ExportTarget::Srt => self.format_srt(entry),
            ExportTarget::Vtt => self.format_vtt(entry),
            ExportTarget::Ass => self.format_ass(entry),
            ExportTarget::Ttml => self.format_ttml(entry),
        }
    }

    /// Produce a complete file header for the given target.
    ///
    /// Returns `None` for formats (like SRT) that have no header.
    #[must_use]
    pub fn file_header(&self, target: &ExportTarget) -> Option<String> {
        match target {
            ExportTarget::Vtt => Some("WEBVTT\n\n".to_string()),
            ExportTarget::Ass => Some(
                "[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\n\n\
                 [V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, \
                 SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, \
                 StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, \
                 Alignment, MarginL, MarginR, MarginV, Encoding\n\
                 Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,\
                 0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n\
                 [Events]\n\
                 Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, \
                 Effect, Text\n"
                    .to_string(),
            ),
            ExportTarget::Ttml => Some(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                 <tt xml:lang=\"en\" xmlns=\"http://www.w3.org/ns/ttml\">\n\
                 <body><div>\n"
                    .to_string(),
            ),
            ExportTarget::Srt => None,
        }
    }

    /// Produce a complete file footer for the given target.
    ///
    /// Returns `None` for formats with no footer.
    #[must_use]
    pub fn file_footer(&self, target: &ExportTarget) -> Option<String> {
        match target {
            ExportTarget::Ttml => Some("</div></body></tt>\n".to_string()),
            _ => None,
        }
    }

    /// Export an entire sequence of entries to a single `String`.
    #[must_use]
    pub fn export_all(&self, entries: &[ExportEntry], target: &ExportTarget) -> String {
        let mut out = String::new();
        if let Some(header) = self.file_header(target) {
            out.push_str(&header);
        }
        for entry in entries {
            out.push_str(&self.format_entry(entry, target));
        }
        if let Some(footer) = self.file_footer(target) {
            out.push_str(&footer);
        }
        out
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn format_srt(&self, entry: &ExportEntry) -> String {
        format!(
            "{}\n{} --> {}\n{}\n\n",
            entry.sequence,
            ms_to_srt_ts(entry.start_ms),
            ms_to_srt_ts(entry.end_ms),
            entry.text,
        )
    }

    fn format_vtt(&self, entry: &ExportEntry) -> String {
        format!(
            "{} --> {}\n{}\n\n",
            ms_to_vtt_ts(entry.start_ms),
            ms_to_vtt_ts(entry.end_ms),
            entry.text,
        )
    }

    fn format_ass(&self, entry: &ExportEntry) -> String {
        format!(
            "Dialogue: 0,{},{},Default,,0,0,0,,{}\n",
            ms_to_ass_ts(entry.start_ms),
            ms_to_ass_ts(entry.end_ms),
            entry.text,
        )
    }

    fn format_ttml(&self, entry: &ExportEntry) -> String {
        format!(
            "  <p begin=\"{}\" end=\"{}\">{}</p>\n",
            ms_to_vtt_ts(entry.start_ms),
            ms_to_vtt_ts(entry.end_ms),
            entry.text,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry() -> ExportEntry {
        ExportEntry::new(1, 1_000, 4_000, "Hello, world!")
    }

    #[test]
    fn test_srt_sequence_number() {
        let exporter = SubtitleExporter::new();
        let out = exporter.format_entry(&sample_entry(), &ExportTarget::Srt);
        assert!(out.starts_with("1\n"));
    }

    #[test]
    fn test_srt_timestamp_format() {
        let exporter = SubtitleExporter::new();
        let out = exporter.format_entry(&sample_entry(), &ExportTarget::Srt);
        assert!(out.contains("00:00:01,000 --> 00:00:04,000"));
    }

    #[test]
    fn test_srt_text_present() {
        let exporter = SubtitleExporter::new();
        let out = exporter.format_entry(&sample_entry(), &ExportTarget::Srt);
        assert!(out.contains("Hello, world!"));
    }

    #[test]
    fn test_vtt_timestamp_format() {
        let exporter = SubtitleExporter::new();
        let out = exporter.format_entry(&sample_entry(), &ExportTarget::Vtt);
        assert!(out.contains("00:00:01.000 --> 00:00:04.000"));
    }

    #[test]
    fn test_vtt_no_sequence_number() {
        let exporter = SubtitleExporter::new();
        let out = exporter.format_entry(&sample_entry(), &ExportTarget::Vtt);
        // VTT entries don't start with a bare number line.
        assert!(!out.starts_with("1\n"));
    }

    #[test]
    fn test_ass_format_contains_dialogue() {
        let exporter = SubtitleExporter::new();
        let out = exporter.format_entry(&sample_entry(), &ExportTarget::Ass);
        assert!(out.starts_with("Dialogue:"));
    }

    #[test]
    fn test_ass_timestamp_format() {
        let exporter = SubtitleExporter::new();
        let out = exporter.format_entry(&sample_entry(), &ExportTarget::Ass);
        // ASS uses H:MM:SS.cc
        assert!(out.contains("0:00:01.00"));
        assert!(out.contains("0:00:04.00"));
    }

    #[test]
    fn test_ttml_format() {
        let exporter = SubtitleExporter::new();
        let out = exporter.format_entry(&sample_entry(), &ExportTarget::Ttml);
        assert!(out.contains("<p begin="));
        assert!(out.contains("Hello, world!"));
        assert!(out.contains("</p>"));
    }

    #[test]
    fn test_vtt_header() {
        let exporter = SubtitleExporter::new();
        let hdr = exporter
            .file_header(&ExportTarget::Vtt)
            .expect("should succeed in test");
        assert!(hdr.starts_with("WEBVTT"));
    }

    #[test]
    fn test_srt_no_header() {
        let exporter = SubtitleExporter::new();
        assert!(exporter.file_header(&ExportTarget::Srt).is_none());
    }

    #[test]
    fn test_ttml_has_header_and_footer() {
        let exporter = SubtitleExporter::new();
        assert!(exporter.file_header(&ExportTarget::Ttml).is_some());
        assert!(exporter.file_footer(&ExportTarget::Ttml).is_some());
    }

    #[test]
    fn test_export_all_srt() {
        let entries = vec![
            ExportEntry::new(1, 0, 2000, "First"),
            ExportEntry::new(2, 3000, 5000, "Second"),
        ];
        let exporter = SubtitleExporter::new();
        let out = exporter.export_all(&entries, &ExportTarget::Srt);
        assert!(out.contains("1\n"));
        assert!(out.contains("2\n"));
        assert!(out.contains("First"));
        assert!(out.contains("Second"));
    }

    #[test]
    fn test_export_target_extension() {
        assert_eq!(ExportTarget::Srt.extension(), "srt");
        assert_eq!(ExportTarget::Vtt.extension(), "vtt");
        assert_eq!(ExportTarget::Ass.extension(), "ass");
        assert_eq!(ExportTarget::Ttml.extension(), "ttml");
    }

    #[test]
    fn test_export_target_mime_type() {
        assert_eq!(ExportTarget::Vtt.mime_type(), "text/vtt");
        assert_eq!(ExportTarget::Ttml.mime_type(), "application/ttml+xml");
    }

    #[test]
    fn test_ms_to_srt_ts_zero() {
        assert_eq!(ms_to_srt_ts(0), "00:00:00,000");
    }

    #[test]
    fn test_ms_to_vtt_ts_hours() {
        // 3_661_500 ms = 1h 1m 1.5s
        assert_eq!(ms_to_vtt_ts(3_661_500), "01:01:01.500");
    }
}
