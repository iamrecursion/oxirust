//! EDL metadata and header management.
//!
//! Provides structured access to the metadata that appears in the header
//! section of an EDL file — title, frame-count mode, project name, tape name,
//! and other informational fields.

#![allow(dead_code)]

// ────────────────────────────────────────────────────────────────────────────
// FrameCountMode
// ────────────────────────────────────────────────────────────────────────────

/// The frame-count mode declared in the EDL header (`FCM:` line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameCountMode {
    /// `FCM: DROP FRAME` — 29.97 or 59.94 fps with drop-frame counting.
    DropFrame,
    /// `FCM: NON-DROP FRAME` — any non-drop-frame rate.
    NonDropFrame,
}

impl FrameCountMode {
    /// Return the canonical EDL header string for this mode.
    #[must_use]
    pub fn as_fcm_str(self) -> &'static str {
        match self {
            Self::DropFrame => "DROP FRAME",
            Self::NonDropFrame => "NON-DROP FRAME",
        }
    }

    /// Parse an `FCM:` value string (case-insensitive).
    #[must_use]
    pub fn from_fcm_str(s: &str) -> Option<Self> {
        let upper = s.trim().to_uppercase();
        if upper == "DROP FRAME" || upper == "DF" {
            Some(Self::DropFrame)
        } else if upper.contains("NON") || upper == "NDF" {
            Some(Self::NonDropFrame)
        } else {
            None
        }
    }
}

impl std::fmt::Display for FrameCountMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_fcm_str())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// EdlMetadata
// ────────────────────────────────────────────────────────────────────────────

/// Structured metadata extracted from (or to be written into) an EDL header.
#[derive(Debug, Clone, Default)]
pub struct EdlMetadata {
    /// `TITLE:` field — the project or sequence name.
    pub title: Option<String>,
    /// `FCM:` field — frame-count mode.
    pub frame_count_mode: Option<FrameCountMode>,
    /// Optional source project name (non-standard, found in some NLEs).
    pub project_name: Option<String>,
    /// Optional client / production house name.
    pub client_name: Option<String>,
    /// Optional date string (free-form).
    pub date: Option<String>,
    /// Arbitrary key-value pairs for non-standard header fields.
    pub extra_fields: Vec<(String, String)>,
}

impl EdlMetadata {
    /// Create empty metadata.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the title.
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = Some(title.into());
    }

    /// Set the frame-count mode.
    pub fn set_frame_count_mode(&mut self, mode: FrameCountMode) {
        self.frame_count_mode = Some(mode);
    }

    /// Add an arbitrary header field.
    pub fn add_extra_field(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.extra_fields.push((key.into(), value.into()));
    }

    /// Look up a value in the extra fields by key (case-insensitive).
    #[must_use]
    pub fn get_extra_field(&self, key: &str) -> Option<&str> {
        let upper = key.to_uppercase();
        self.extra_fields
            .iter()
            .find(|(k, _)| k.to_uppercase() == upper)
            .map(|(_, v)| v.as_str())
    }

    /// Render the metadata as EDL header lines.
    #[must_use]
    pub fn to_header_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();

        if let Some(ref t) = self.title {
            lines.push(format!("TITLE: {t}"));
        }
        if let Some(fcm) = self.frame_count_mode {
            lines.push(format!("FCM: {fcm}"));
        }
        if let Some(ref p) = self.project_name {
            lines.push(format!("* PROJECT: {p}"));
        }
        if let Some(ref c) = self.client_name {
            lines.push(format!("* CLIENT: {c}"));
        }
        if let Some(ref d) = self.date {
            lines.push(format!("* DATE: {d}"));
        }
        for (k, v) in &self.extra_fields {
            lines.push(format!("* {k}: {v}"));
        }

        lines
    }

    /// Returns `true` if no metadata fields are set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.title.is_none()
            && self.frame_count_mode.is_none()
            && self.project_name.is_none()
            && self.client_name.is_none()
            && self.date.is_none()
            && self.extra_fields.is_empty()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MetadataParser
// ────────────────────────────────────────────────────────────────────────────

/// Lightweight parser for EDL header lines (stops at the first event line).
#[derive(Debug, Default)]
pub struct MetadataParser;

impl MetadataParser {
    /// Parse the header portion of `text`, returning structured metadata.
    #[must_use]
    pub fn parse(&self, text: &str) -> EdlMetadata {
        let mut meta = EdlMetadata::new();

        for line in text.lines() {
            let trimmed = line.trim();

            // Stop when we reach the first event line (starts with a digit).
            if trimmed.starts_with(|c: char| c.is_ascii_digit()) {
                break;
            }

            if let Some(rest) = trimmed.strip_prefix("TITLE:") {
                meta.set_title(rest.trim());
            } else if let Some(rest) = trimmed.strip_prefix("FCM:") {
                if let Some(mode) = FrameCountMode::from_fcm_str(rest.trim()) {
                    meta.set_frame_count_mode(mode);
                }
            } else if let Some(rest) = trimmed.strip_prefix("* PROJECT:") {
                meta.project_name = Some(rest.trim().to_string());
            } else if let Some(rest) = trimmed.strip_prefix("* CLIENT:") {
                meta.client_name = Some(rest.trim().to_string());
            } else if let Some(rest) = trimmed.strip_prefix("* DATE:") {
                meta.date = Some(rest.trim().to_string());
            } else if let Some(rest) = trimmed.strip_prefix("* ") {
                // Generic key: value.
                if let Some(colon) = rest.find(':') {
                    let key = rest[..colon].trim().to_string();
                    let val = rest[colon + 1..].trim().to_string();
                    meta.add_extra_field(key, val);
                }
            }
        }

        meta
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_count_mode_as_fcm_str() {
        assert_eq!(FrameCountMode::DropFrame.as_fcm_str(), "DROP FRAME");
        assert_eq!(FrameCountMode::NonDropFrame.as_fcm_str(), "NON-DROP FRAME");
    }

    #[test]
    fn test_frame_count_mode_from_fcm_str_drop_frame() {
        assert_eq!(
            FrameCountMode::from_fcm_str("DROP FRAME"),
            Some(FrameCountMode::DropFrame)
        );
        assert_eq!(
            FrameCountMode::from_fcm_str("DF"),
            Some(FrameCountMode::DropFrame)
        );
    }

    #[test]
    fn test_frame_count_mode_from_fcm_str_non_drop() {
        assert_eq!(
            FrameCountMode::from_fcm_str("NON-DROP FRAME"),
            Some(FrameCountMode::NonDropFrame)
        );
        assert_eq!(
            FrameCountMode::from_fcm_str("NDF"),
            Some(FrameCountMode::NonDropFrame)
        );
    }

    #[test]
    fn test_frame_count_mode_from_fcm_str_unknown() {
        assert_eq!(FrameCountMode::from_fcm_str("UNKNOWN"), None);
    }

    #[test]
    fn test_frame_count_mode_display() {
        assert_eq!(FrameCountMode::DropFrame.to_string(), "DROP FRAME");
        assert_eq!(FrameCountMode::NonDropFrame.to_string(), "NON-DROP FRAME");
    }

    #[test]
    fn test_edl_metadata_new_is_empty() {
        let meta = EdlMetadata::new();
        assert!(meta.is_empty());
    }

    #[test]
    fn test_edl_metadata_set_title() {
        let mut meta = EdlMetadata::new();
        meta.set_title("My Sequence");
        assert_eq!(meta.title.as_deref(), Some("My Sequence"));
        assert!(!meta.is_empty());
    }

    #[test]
    fn test_edl_metadata_set_frame_count_mode() {
        let mut meta = EdlMetadata::new();
        meta.set_frame_count_mode(FrameCountMode::DropFrame);
        assert_eq!(meta.frame_count_mode, Some(FrameCountMode::DropFrame));
    }

    #[test]
    fn test_edl_metadata_extra_field_roundtrip() {
        let mut meta = EdlMetadata::new();
        meta.add_extra_field("EDITOR", "Jane Smith");
        assert_eq!(meta.get_extra_field("EDITOR"), Some("Jane Smith"));
        assert_eq!(meta.get_extra_field("editor"), Some("Jane Smith")); // case-insensitive
    }

    #[test]
    fn test_edl_metadata_extra_field_not_found() {
        let meta = EdlMetadata::new();
        assert_eq!(meta.get_extra_field("MISSING"), None);
    }

    #[test]
    fn test_to_header_lines_title_and_fcm() {
        let mut meta = EdlMetadata::new();
        meta.set_title("Test EDL");
        meta.set_frame_count_mode(FrameCountMode::NonDropFrame);
        let lines = meta.to_header_lines();
        assert!(lines.iter().any(|l| l.contains("Test EDL")));
        assert!(lines.iter().any(|l| l.contains("NON-DROP FRAME")));
    }

    #[test]
    fn test_to_header_lines_empty_when_no_fields() {
        let meta = EdlMetadata::new();
        assert!(meta.to_header_lines().is_empty());
    }

    #[test]
    fn test_metadata_parser_extracts_title() {
        let text = "TITLE: Parsed Title\nFCM: NON-DROP FRAME\n\n001  AX  V C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let parser = MetadataParser::default();
        let meta = parser.parse(text);
        assert_eq!(meta.title.as_deref(), Some("Parsed Title"));
    }

    #[test]
    fn test_metadata_parser_extracts_fcm() {
        let text = "TITLE: T\nFCM: DROP FRAME\n";
        let parser = MetadataParser::default();
        let meta = parser.parse(text);
        assert_eq!(meta.frame_count_mode, Some(FrameCountMode::DropFrame));
    }

    #[test]
    fn test_metadata_parser_stops_at_event_line() {
        // The "002" line should NOT be treated as a metadata field.
        let text = "TITLE: T\n001  AX  V C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let parser = MetadataParser::default();
        let meta = parser.parse(text);
        assert!(meta.extra_fields.is_empty());
    }

    #[test]
    fn test_metadata_parser_project_and_client() {
        let text = "TITLE: X\n* PROJECT: MyProject\n* CLIENT: Acme Corp\n";
        let parser = MetadataParser::default();
        let meta = parser.parse(text);
        assert_eq!(meta.project_name.as_deref(), Some("MyProject"));
        assert_eq!(meta.client_name.as_deref(), Some("Acme Corp"));
    }

    #[test]
    fn test_metadata_parser_date_field() {
        let text = "TITLE: X\n* DATE: 2026-03-02\n";
        let parser = MetadataParser::default();
        let meta = parser.parse(text);
        assert_eq!(meta.date.as_deref(), Some("2026-03-02"));
    }
}
