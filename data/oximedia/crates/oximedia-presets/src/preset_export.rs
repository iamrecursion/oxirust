//! Preset export to various textual formats.
//!
//! Converts internal preset parameter maps into command-line strings,
//! INI-style configuration blocks, or key=value lists that external
//! tools (ffmpeg, x264, etc.) can consume directly.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fmt;

// ── ExportFormat ───────────────────────────────────────────────────────────

/// Supported output formats for preset export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Key=value pairs, one per line.
    KeyValue,
    /// INI-style with `[section]` headers.
    Ini,
    /// Single-line CLI flags (`-key value`).
    CliFlags,
    /// Comma-separated `key:value` pairs (FFmpeg `-x264-params` style).
    ColonDelimited,
}

impl fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyValue => write!(f, "key=value"),
            Self::Ini => write!(f, "ini"),
            Self::CliFlags => write!(f, "cli"),
            Self::ColonDelimited => write!(f, "colon-delimited"),
        }
    }
}

// ── ExportEntry ────────────────────────────────────────────────────────────

/// A single exported parameter value.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportEntry {
    /// Parameter key.
    pub key: String,
    /// Parameter value as a string.
    pub value: String,
    /// Optional section/group name (used by INI format).
    pub section: Option<String>,
}

impl ExportEntry {
    /// Create a new entry with no section.
    #[must_use]
    pub fn new(key: &str, value: &str) -> Self {
        Self {
            key: key.to_string(),
            value: value.to_string(),
            section: None,
        }
    }

    /// Create an entry belonging to a specific section.
    #[must_use]
    pub fn with_section(key: &str, value: &str, section: &str) -> Self {
        Self {
            key: key.to_string(),
            value: value.to_string(),
            section: Some(section.to_string()),
        }
    }

    /// Format the entry for a given export format, ignoring section grouping.
    #[must_use]
    pub fn format_for(&self, fmt: ExportFormat) -> String {
        match fmt {
            ExportFormat::KeyValue => format!("{}={}", self.key, self.value),
            ExportFormat::Ini => format!("{}={}", self.key, self.value),
            ExportFormat::CliFlags => format!("-{} {}", self.key, self.value),
            ExportFormat::ColonDelimited => format!("{}:{}", self.key, self.value),
        }
    }
}

// ── PresetExporter ─────────────────────────────────────────────────────────

/// Collects [`ExportEntry`] items and renders them in a chosen
/// [`ExportFormat`].
#[derive(Debug, Clone)]
pub struct PresetExporter {
    /// Name of the preset being exported.
    pub name: String,
    /// Entries to export (insertion-order is *not* guaranteed; output is
    /// sorted by key for determinism).
    entries: Vec<ExportEntry>,
    /// Target export format.
    format: ExportFormat,
}

impl PresetExporter {
    /// Create a new exporter.
    #[must_use]
    pub fn new(name: &str, format: ExportFormat) -> Self {
        Self {
            name: name.to_string(),
            entries: Vec::new(),
            format,
        }
    }

    /// Add an entry.
    pub fn add(&mut self, entry: ExportEntry) {
        self.entries.push(entry);
    }

    /// Builder-style entry addition.
    #[must_use]
    pub fn with_entry(mut self, entry: ExportEntry) -> Self {
        self.add(entry);
        self
    }

    /// Number of entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Change the target format.
    pub fn set_format(&mut self, format: ExportFormat) {
        self.format = format;
    }

    /// The current format.
    #[must_use]
    pub fn format(&self) -> ExportFormat {
        self.format
    }

    /// Render the entries to a string in the current format.
    ///
    /// Entries are sorted by key for deterministic output. For INI format,
    /// entries are additionally grouped by section.
    #[must_use]
    pub fn render(&self) -> String {
        match self.format {
            ExportFormat::KeyValue => self.render_key_value(),
            ExportFormat::Ini => self.render_ini(),
            ExportFormat::CliFlags => self.render_cli(),
            ExportFormat::ColonDelimited => self.render_colon(),
        }
    }

    /// Render as sorted key=value lines.
    fn render_key_value(&self) -> String {
        let mut sorted: Vec<&ExportEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| a.key.cmp(&b.key));
        sorted
            .iter()
            .map(|e| e.format_for(ExportFormat::KeyValue))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Render as INI sections.
    fn render_ini(&self) -> String {
        // Group by section.
        let mut sections: BTreeMap<String, Vec<&ExportEntry>> = BTreeMap::new();
        for e in &self.entries {
            let sec = e.section.clone().unwrap_or_else(|| "general".to_string());
            sections.entry(sec).or_default().push(e);
        }
        let mut out = String::new();
        for (sec, entries) in &sections {
            out.push_str(&format!("[{}]\n", sec));
            let mut sorted = entries.clone();
            sorted.sort_by(|a, b| a.key.cmp(&b.key));
            for e in sorted {
                out.push_str(&format!("{}={}\n", e.key, e.value));
            }
            out.push('\n');
        }
        // Trim trailing whitespace.
        out.trim_end().to_string()
    }

    /// Render as a single-line CLI flag string.
    fn render_cli(&self) -> String {
        let mut sorted: Vec<&ExportEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| a.key.cmp(&b.key));
        sorted
            .iter()
            .map(|e| e.format_for(ExportFormat::CliFlags))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Render as colon-delimited pairs joined by commas.
    fn render_colon(&self) -> String {
        let mut sorted: Vec<&ExportEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| a.key.cmp(&b.key));
        sorted
            .iter()
            .map(|e| e.format_for(ExportFormat::ColonDelimited))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Return all unique section names across entries, sorted.
    #[must_use]
    pub fn sections(&self) -> Vec<String> {
        let mut secs: Vec<String> = self
            .entries
            .iter()
            .filter_map(|e| e.section.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        secs.sort();
        secs
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_exporter(fmt: ExportFormat) -> PresetExporter {
        PresetExporter::new("test-preset", fmt)
            .with_entry(ExportEntry::new("bitrate", "5000000"))
            .with_entry(ExportEntry::new("codec", "h264"))
            .with_entry(ExportEntry::new("crf", "23"))
    }

    // ── ExportEntry ──

    #[test]
    fn test_entry_key_value_format() {
        let e = ExportEntry::new("fps", "30");
        assert_eq!(e.format_for(ExportFormat::KeyValue), "fps=30");
    }

    #[test]
    fn test_entry_cli_format() {
        let e = ExportEntry::new("fps", "30");
        assert_eq!(e.format_for(ExportFormat::CliFlags), "-fps 30");
    }

    #[test]
    fn test_entry_colon_format() {
        let e = ExportEntry::new("fps", "30");
        assert_eq!(e.format_for(ExportFormat::ColonDelimited), "fps:30");
    }

    #[test]
    fn test_entry_with_section() {
        let e = ExportEntry::with_section("width", "1920", "video");
        assert_eq!(e.section, Some("video".to_string()));
    }

    // ── PresetExporter key-value ──

    #[test]
    fn test_render_key_value_sorted() {
        let ex = sample_exporter(ExportFormat::KeyValue);
        let rendered = ex.render();
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines.len(), 3);
        // sorted: bitrate, codec, crf
        assert_eq!(lines[0], "bitrate=5000000");
        assert_eq!(lines[1], "codec=h264");
        assert_eq!(lines[2], "crf=23");
    }

    // ── PresetExporter CLI ──

    #[test]
    fn test_render_cli_single_line() {
        let ex = sample_exporter(ExportFormat::CliFlags);
        let rendered = ex.render();
        assert!(!rendered.contains('\n'));
        assert!(rendered.contains("-bitrate 5000000"));
        assert!(rendered.contains("-codec h264"));
    }

    // ── PresetExporter colon-delimited ──

    #[test]
    fn test_render_colon_delimited() {
        let ex = sample_exporter(ExportFormat::ColonDelimited);
        let rendered = ex.render();
        assert!(rendered.contains("bitrate:5000000"));
        assert!(rendered.contains("codec:h264"));
        // comma separated
        assert!(rendered.contains(','));
    }

    // ── PresetExporter INI ──

    #[test]
    fn test_render_ini_default_section() {
        let ex = sample_exporter(ExportFormat::Ini);
        let rendered = ex.render();
        assert!(rendered.contains("[general]"));
        assert!(rendered.contains("bitrate=5000000"));
    }

    #[test]
    fn test_render_ini_custom_sections() {
        let mut ex = PresetExporter::new("multi-section", ExportFormat::Ini);
        ex.add(ExportEntry::with_section("width", "1920", "video"));
        ex.add(ExportEntry::with_section("height", "1080", "video"));
        ex.add(ExportEntry::with_section("channels", "2", "audio"));
        let rendered = ex.render();
        assert!(rendered.contains("[video]"));
        assert!(rendered.contains("[audio]"));
        assert!(rendered.contains("width=1920"));
        assert!(rendered.contains("channels=2"));
    }

    // ── Misc ──

    #[test]
    fn test_entry_count() {
        let ex = sample_exporter(ExportFormat::KeyValue);
        assert_eq!(ex.entry_count(), 3);
    }

    #[test]
    fn test_set_format() {
        let mut ex = sample_exporter(ExportFormat::KeyValue);
        ex.set_format(ExportFormat::CliFlags);
        assert_eq!(ex.format(), ExportFormat::CliFlags);
    }

    #[test]
    fn test_sections_list() {
        let mut ex = PresetExporter::new("sec", ExportFormat::Ini);
        ex.add(ExportEntry::with_section("a", "1", "video"));
        ex.add(ExportEntry::with_section("b", "2", "audio"));
        let secs = ex.sections();
        assert_eq!(secs, vec!["audio", "video"]);
    }

    #[test]
    fn test_export_format_display() {
        assert_eq!(ExportFormat::KeyValue.to_string(), "key=value");
        assert_eq!(ExportFormat::Ini.to_string(), "ini");
        assert_eq!(ExportFormat::CliFlags.to_string(), "cli");
        assert_eq!(ExportFormat::ColonDelimited.to_string(), "colon-delimited");
    }
}
