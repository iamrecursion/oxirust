//! Caption export: profile selection, SRT/VTT serialisation, and entry counting.

#![allow(dead_code)]

/// Available export profiles for caption output files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportProfile {
    /// `SubRip` Text format (.srt).
    Srt,
    /// Web Video Text Tracks format (.vtt).
    Vtt,
    /// Timed Text Markup Language (.ttml).
    Ttml,
    /// `SubStation` Alpha (.ssa).
    Ssa,
}

impl ExportProfile {
    /// Return the canonical file extension for this profile.
    #[must_use]
    pub fn file_extension(&self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Ttml => "ttml",
            Self::Ssa => "ssa",
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Srt => "SubRip Text",
            Self::Vtt => "WebVTT",
            Self::Ttml => "TTML",
            Self::Ssa => "SubStation Alpha",
        }
    }

    /// Returns `true` if this profile produces a plain-text output.
    #[must_use]
    pub fn is_text_format(&self) -> bool {
        matches!(self, Self::Srt | Self::Vtt | Self::Ssa)
    }

    /// Returns `true` if this profile is an XML-based format.
    #[must_use]
    pub fn is_xml_format(&self) -> bool {
        matches!(self, Self::Ttml)
    }
}

/// A caption entry ready for export.
#[derive(Debug, Clone)]
pub struct ExportEntry {
    /// 1-based index.
    pub index: usize,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Caption text (may contain newlines for multi-line captions).
    pub text: String,
}

impl ExportEntry {
    /// Create a new export entry.
    #[must_use]
    pub fn new(index: usize, start_ms: u64, end_ms: u64, text: impl Into<String>) -> Self {
        Self {
            index,
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    /// Format `ms` as `HH:MM:SS,mmm` (SRT timestamp).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    fn srt_timestamp(ms: u64) -> String {
        let h = ms / 3_600_000;
        let m = (ms % 3_600_000) / 60_000;
        let s = (ms % 60_000) / 1_000;
        let frac = ms % 1_000;
        format!("{h:02}:{m:02}:{s:02},{frac:03}")
    }

    /// Format `ms` as `HH:MM:SS.mmm` (`WebVTT` timestamp).
    #[must_use]
    fn vtt_timestamp(ms: u64) -> String {
        let h = ms / 3_600_000;
        let m = (ms % 3_600_000) / 60_000;
        let s = (ms % 60_000) / 1_000;
        let frac = ms % 1_000;
        format!("{h:02}:{m:02}:{s:02}.{frac:03}")
    }

    /// Render this entry as an SRT block.
    #[must_use]
    pub fn to_srt_block(&self) -> String {
        format!(
            "{}\n{} --> {}\n{}\n",
            self.index,
            Self::srt_timestamp(self.start_ms),
            Self::srt_timestamp(self.end_ms),
            self.text
        )
    }

    /// Render this entry as a `WebVTT` cue block.
    #[must_use]
    pub fn to_vtt_cue(&self) -> String {
        format!(
            "{} --> {}\n{}\n",
            Self::vtt_timestamp(self.start_ms),
            Self::vtt_timestamp(self.end_ms),
            self.text
        )
    }
}

/// Exports a collection of caption entries to various text formats.
#[derive(Debug, Clone)]
pub struct CaptionExporter {
    entries: Vec<ExportEntry>,
}

impl CaptionExporter {
    /// Create a new exporter from a list of entries.
    ///
    /// Entries should be ordered by `start_ms`.
    #[must_use]
    pub fn new(entries: Vec<ExportEntry>) -> Self {
        Self { entries }
    }

    /// Return the number of caption entries held by this exporter.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Return a reference to the internal entries.
    #[must_use]
    pub fn entries(&self) -> &[ExportEntry] {
        &self.entries
    }

    /// Export all entries to SRT format.
    ///
    /// Blocks are separated by a blank line as required by the specification.
    #[must_use]
    pub fn export_to_srt(&self) -> String {
        let mut out = String::new();
        for entry in &self.entries {
            out.push_str(&entry.to_srt_block());
            out.push('\n');
        }
        out
    }

    /// Export all entries to `WebVTT` format.
    ///
    /// The mandatory `WEBVTT` header line is prepended automatically.
    #[must_use]
    pub fn export_to_vtt(&self) -> String {
        let mut out = String::from("WEBVTT\n\n");
        for entry in &self.entries {
            out.push_str(&entry.to_vtt_cue());
            out.push('\n');
        }
        out
    }

    /// Export all entries to a minimal TTML skeleton.
    ///
    /// The output is a valid but minimal TTML document. Proper TTML authoring
    /// tools should post-process this for production use.
    #[must_use]
    pub fn export_to_ttml(&self) -> String {
        let mut body = String::new();
        for entry in &self.entries {
            let start = ttml_timestamp(entry.start_ms);
            let end = ttml_timestamp(entry.end_ms);
            body.push_str(&format!(
                "      <p begin=\"{}\" end=\"{}\">{}</p>\n",
                start,
                end,
                xml_escape(&entry.text)
            ));
        }
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <tt xmlns=\"http://www.w3.org/ns/ttml\">\n\
               <body>\n\
                 <div>\n\
             {body}\
                 </div>\n\
               </body>\n\
             </tt>\n"
        )
    }

    /// Dispatch export to the requested profile.
    #[must_use]
    pub fn export(&self, profile: ExportProfile) -> String {
        match profile {
            ExportProfile::Srt => self.export_to_srt(),
            ExportProfile::Vtt => self.export_to_vtt(),
            ExportProfile::Ttml => self.export_to_ttml(),
            ExportProfile::Ssa => self.export_to_ssa_stub(),
        }
    }

    /// Minimal SSA stub (header + events).
    #[must_use]
    fn export_to_ssa_stub(&self) -> String {
        let mut out = String::from(
            "[Script Info]\nScriptType: v4.00\n\n[Events]\nFormat: Start, End, Text\n",
        );
        for entry in &self.entries {
            let start = ssa_timestamp(entry.start_ms);
            let end = ssa_timestamp(entry.end_ms);
            out.push_str(&format!("Dialogue: {start},{end},{}\n", entry.text));
        }
        out
    }
}

/// Format ms as `H:MM:SS.cc` (SSA/ASS timestamp, centiseconds).
#[allow(clippy::cast_precision_loss)]
fn ssa_timestamp(ms: u64) -> String {
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1_000;
    let cs = (ms % 1_000) / 10;
    format!("{h}:{m:02}:{s:02}.{cs:02}")
}

/// Format ms as `HH:MM:SS.mmm` (TTML timestamp).
fn ttml_timestamp(ms: u64) -> String {
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1_000;
    let frac = ms % 1_000;
    format!("{h:02}:{m:02}:{s:02}.{frac:03}")
}

/// Minimal XML character escaping for TTML body text.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ─── unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_exporter() -> CaptionExporter {
        CaptionExporter::new(vec![
            ExportEntry::new(1, 0, 3000, "Hello world"),
            ExportEntry::new(2, 5000, 8000, "Second line"),
            ExportEntry::new(3, 10000, 12500, "Third caption"),
        ])
    }

    // 1 — ExportProfile::file_extension
    #[test]
    fn test_file_extension() {
        assert_eq!(ExportProfile::Srt.file_extension(), "srt");
        assert_eq!(ExportProfile::Vtt.file_extension(), "vtt");
        assert_eq!(ExportProfile::Ttml.file_extension(), "ttml");
        assert_eq!(ExportProfile::Ssa.file_extension(), "ssa");
    }

    // 2 — ExportProfile::is_text_format
    #[test]
    fn test_is_text_format() {
        assert!(ExportProfile::Srt.is_text_format());
        assert!(ExportProfile::Vtt.is_text_format());
        assert!(!ExportProfile::Ttml.is_text_format());
    }

    // 3 — ExportProfile::is_xml_format
    #[test]
    fn test_is_xml_format() {
        assert!(ExportProfile::Ttml.is_xml_format());
        assert!(!ExportProfile::Srt.is_xml_format());
    }

    // 4 — ExportProfile::label
    #[test]
    fn test_export_profile_label() {
        assert_eq!(ExportProfile::Srt.label(), "SubRip Text");
        assert_eq!(ExportProfile::Vtt.label(), "WebVTT");
        assert_eq!(ExportProfile::Ttml.label(), "TTML");
        assert_eq!(ExportProfile::Ssa.label(), "SubStation Alpha");
    }

    // 5 — entry_count
    #[test]
    fn test_entry_count() {
        let exp = sample_exporter();
        assert_eq!(exp.entry_count(), 3);
    }

    // 6 — export_to_srt contains index and timestamp
    #[test]
    fn test_export_to_srt_contains_index() {
        let exp = sample_exporter();
        let srt = exp.export_to_srt();
        assert!(srt.contains("1\n"));
        assert!(srt.contains("00:00:00,000 --> 00:00:03,000"));
        assert!(srt.contains("Hello world"));
    }

    // 7 — export_to_srt multi-entry
    #[test]
    fn test_export_to_srt_multi_entry() {
        let exp = sample_exporter();
        let srt = exp.export_to_srt();
        assert!(srt.contains("2\n"));
        assert!(srt.contains("00:00:05,000 --> 00:00:08,000"));
        assert!(srt.contains("Second line"));
    }

    // 8 — export_to_vtt starts with WEBVTT header
    #[test]
    fn test_export_to_vtt_header() {
        let exp = sample_exporter();
        let vtt = exp.export_to_vtt();
        assert!(vtt.starts_with("WEBVTT\n\n"));
    }

    // 9 — export_to_vtt uses dot separator (not comma)
    #[test]
    fn test_export_to_vtt_dot_separator() {
        let exp = sample_exporter();
        let vtt = exp.export_to_vtt();
        assert!(vtt.contains("00:00:00.000 --> 00:00:03.000"));
    }

    // 10 — export_to_vtt contains text
    #[test]
    fn test_export_to_vtt_contains_text() {
        let exp = sample_exporter();
        let vtt = exp.export_to_vtt();
        assert!(vtt.contains("Hello world"));
        assert!(vtt.contains("Third caption"));
    }

    // 11 — export_to_ttml contains XML declaration
    #[test]
    fn test_export_to_ttml_xml_decl() {
        let exp = sample_exporter();
        let ttml = exp.export_to_ttml();
        assert!(ttml.starts_with("<?xml version=\"1.0\""));
        assert!(ttml.contains("<tt xmlns="));
    }

    // 12 — export_to_ttml contains caption text
    #[test]
    fn test_export_to_ttml_text() {
        let exp = sample_exporter();
        let ttml = exp.export_to_ttml();
        assert!(ttml.contains("Hello world"));
    }

    // 13 — xml_escape escapes special characters
    #[test]
    fn test_xml_escape() {
        let s = xml_escape("A & B < C > D \"E\"");
        assert_eq!(s, "A &amp; B &lt; C &gt; D &quot;E&quot;");
    }

    // 14 — export dispatch to Srt equals export_to_srt
    #[test]
    fn test_export_dispatch_srt() {
        let exp = sample_exporter();
        assert_eq!(exp.export(ExportProfile::Srt), exp.export_to_srt());
    }

    // 15 — export dispatch to Vtt equals export_to_vtt
    #[test]
    fn test_export_dispatch_vtt() {
        let exp = sample_exporter();
        assert_eq!(exp.export(ExportProfile::Vtt), exp.export_to_vtt());
    }
}
