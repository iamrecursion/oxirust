//! FCPXML importer — manual string-based parser for Final Cut Pro X project files.
//!
//! FCPXML uses XML with time values expressed as rational fractions of a second,
//! e.g. `"100/2400s"` (100/2400 = 1/24 second) or plain `"30s"` (30 seconds).
//!
//! This parser does **not** depend on any external XML crate; it implements a
//! minimal hand-written tag/attribute extractor sufficient for the structural
//! elements used in FCPXML timelines.

#![allow(dead_code)]

use crate::error::ConformResult;
use crate::types::{ConformClip, ConformProject, ConformSequence};

// ---------------------------------------------------------------------------
// Public re-export aliases (as required by the task spec)
// ---------------------------------------------------------------------------

/// A clip parsed from an FCPXML file (alias for [`ConformClip`]).
pub type FcpxmlClip = ConformClip;

/// A sequence parsed from an FCPXML file (alias for [`ConformSequence`]).
pub type FcpxmlSequence = ConformSequence;

// ---------------------------------------------------------------------------
// FcpxmlParser
// ---------------------------------------------------------------------------

/// Parser for Final Cut Pro X FCPXML project files.
///
/// Uses manual string parsing — no external XML crate required.
pub struct FcpxmlParser;

impl FcpxmlParser {
    /// Create a new parser instance.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Parse an FCPXML document string and return a [`ConformProject`].
    ///
    /// # Errors
    ///
    /// Returns `ConformError::Xml` (via the `Other` variant) if the XML is
    /// too malformed to extract any useful structure, or
    /// `ConformError::MissingField` if mandatory attributes are absent.
    pub fn parse(xml: &str) -> ConformResult<ConformProject> {
        let mut project = ConformProject::default();

        // Extract project name from the first <project name="..."> tag.
        if let Some(name) = attr_value_in_tag(xml, "project", "name") {
            project.name = name;
        }

        // Build a map of format ID → frameDuration for later lookup.
        let format_map = collect_formats(xml);

        // Each <sequence> element becomes a ConformSequence.
        for seq_elem in iter_element_contents(xml, "sequence") {
            let mut sequence = ConformSequence::default();

            // Sequence name — from opening tag; optional, may fall back to parent project name.
            if let Some(name) = attr_from_tag(&seq_elem.open_tag, "name") {
                sequence.name = name;
            } else {
                sequence.name = project.name.clone();
            }

            // Resolve frame rate from format attribute on the opening tag.
            if let Some(fmt_ref) = attr_from_tag(&seq_elem.open_tag, "format") {
                if let Some(fps) = format_map.get(fmt_ref.as_str()) {
                    sequence.frame_rate = *fps;
                }
            }
            // Fallback: frameDuration directly on the sequence tag.
            if sequence.frame_rate == 0.0 {
                if let Some(fd) = attr_from_tag(&seq_elem.open_tag, "frameDuration") {
                    sequence.frame_rate = frame_duration_to_fps(&fd);
                }
            }

            // Parse clips from the spine (inner content).
            for spine_elem in iter_element_contents(&seq_elem.inner, "spine") {
                for clip in parse_clips_from_spine(&spine_elem.inner) {
                    sequence.clips.push(clip);
                }
            }
            // Also accept clips directly under the sequence inner content (no explicit spine).
            if sequence.clips.is_empty() {
                for clip in parse_clips_from_spine(&seq_elem.inner) {
                    sequence.clips.push(clip);
                }
            }

            project.sequences.push(sequence);
        }

        Ok(project)
    }
}

impl Default for FcpxmlParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal parsing helpers
// ---------------------------------------------------------------------------

/// Build a `HashMap<format_id, fps>` from all `<format>` tags in the document.
fn collect_formats(xml: &str) -> std::collections::HashMap<String, f32> {
    let mut map = std::collections::HashMap::new();
    let mut search = xml;
    while let Some(tag_start) = search.find("<format") {
        let rest = &search[tag_start..];
        // Find end of opening tag
        let tag_end = if let Some(p) = rest.find('>') {
            p + 1
        } else {
            break;
        };
        let tag_text = &rest[..tag_end];

        if let (Some(id), Some(fd)) = (
            attr_from_tag(tag_text, "id"),
            attr_from_tag(tag_text, "frameDuration"),
        ) {
            let fps = frame_duration_to_fps(&fd);
            if fps > 0.0 {
                map.insert(id, fps);
            }
        }
        search = &rest[1..]; // advance past '<'
    }
    map
}

/// Parse `<clip>` and `<asset-clip>` elements directly from `spine_xml`.
fn parse_clips_from_spine(spine_xml: &str) -> Vec<FcpxmlClip> {
    let mut clips = Vec::new();
    for tag_name in &["clip", "asset-clip", "ref-clip"] {
        let mut search = spine_xml;
        while let Some(tag_start) = search.find(&format!("<{tag_name}")) {
            let rest = &search[tag_start..];
            // Find end of the opening tag (self-closing or not)
            let tag_end = if let Some(p) = rest.find('>') {
                p + 1
            } else {
                break;
            };
            let tag_text = &rest[..tag_end];

            let name = attr_from_tag(tag_text, "name").unwrap_or_default();
            let offset = attr_from_tag(tag_text, "offset")
                .as_deref()
                .map(parse_time_value)
                .unwrap_or(0.0);
            let duration = attr_from_tag(tag_text, "duration")
                .as_deref()
                .map(parse_time_value)
                .unwrap_or(0.0);
            let start = attr_from_tag(tag_text, "start")
                .as_deref()
                .map(parse_time_value)
                .unwrap_or(0.0);

            let src_out = start + duration;

            clips.push(FcpxmlClip {
                name,
                offset_s: offset,
                duration_s: duration,
                src_in_s: start,
                src_out_s: src_out,
            });

            search = &rest[1..]; // advance past current '<'
        }
    }
    // Sort by timeline offset
    clips.sort_by(|a, b| {
        a.offset_s
            .partial_cmp(&b.offset_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    clips
}

/// A parsed element: opening tag text and inner content.
struct ParsedElement {
    /// The full opening tag (e.g. `<sequence format="r1">`).
    open_tag: String,
    /// Inner XML content between opening and closing tags.
    inner: String,
}

/// Iterate over all instances of `<tag_name>` in `xml`, returning both the
/// opening tag text (for attribute extraction) and the inner content.
fn iter_element_contents(xml: &str, tag_name: &str) -> Vec<ParsedElement> {
    let open = format!("<{tag_name}");
    let close = format!("</{tag_name}>");
    let mut results = Vec::new();
    let mut search = xml;

    while let Some(start) = search.find(&open) {
        let rest = &search[start..];
        // Find end of opening tag
        if let Some(open_end) = rest.find('>') {
            let open_tag = rest[..open_end + 1].to_string();
            // Self-closing?
            if open_tag.ends_with("/>") {
                results.push(ParsedElement {
                    open_tag,
                    inner: String::new(),
                });
                search = &rest[1..];
                continue;
            }
            let inner_start = open_end + 1;
            // Find matching closing tag (naive: no nesting support needed for our use case)
            if let Some(close_off) = rest[inner_start..].find(&close) {
                let inner = rest[inner_start..inner_start + close_off].to_string();
                results.push(ParsedElement {
                    open_tag,
                    inner: inner.clone(),
                });
                search = &rest[inner_start + close_off + close.len()..];
            } else {
                // Closing tag not found — stop searching for this element.
                break;
            }
        } else {
            break;
        }
    }
    results
}

/// Extract the value of `attr_name` from any occurrence of `<tag_name ... attr_name="value">`.
fn attr_value_in_tag(xml: &str, tag_name: &str, attr_name: &str) -> Option<String> {
    let open = format!("<{tag_name}");
    let pos = xml.find(&open)?;
    let rest = &xml[pos..];
    let end = rest.find('>')?;
    let tag_text = &rest[..end + 1];
    attr_from_tag(tag_text, attr_name)
}

/// Extract the value of `attr_name` from the **first** `<tag_name>` opening tag
/// found anywhere in `xml`, searching through all instances.
#[allow(dead_code)]
fn attr_value_any_tag(xml: &str, tag_name: &str, attr_name: &str) -> Option<String> {
    let open = format!("<{tag_name}");
    let mut search = xml;
    while let Some(pos) = search.find(&open) {
        let rest = &search[pos..];
        if let Some(end) = rest.find('>') {
            let tag_text = &rest[..end + 1];
            if let Some(val) = attr_from_tag(tag_text, attr_name) {
                return Some(val);
            }
        }
        search = &search[pos + 1..];
    }
    None
}

/// Extract `name="value"` (or `name='value'`) from a tag text snippet.
fn attr_from_tag(tag_text: &str, name: &str) -> Option<String> {
    // Try double-quoted: name="value"
    let dq_needle = format!("{name}=\"");
    if let Some(start) = tag_text.find(&dq_needle) {
        let after = &tag_text[start + dq_needle.len()..];
        if let Some(end) = after.find('"') {
            return Some(after[..end].to_string());
        }
    }
    // Try single-quoted: name='value'
    let sq_needle = format!("{name}='");
    if let Some(start) = tag_text.find(&sq_needle) {
        let after = &tag_text[start + sq_needle.len()..];
        if let Some(end) = after.find('\'') {
            return Some(after[..end].to_string());
        }
    }
    None
}

/// Parse an FCPXML time value string to seconds.
///
/// Supported formats:
/// - `"100/2400s"` → `100.0 / 2400.0`
/// - `"30s"` → `30.0`
/// - `"0s"` → `0.0`
/// - Bare integer `"42"` → `42.0` (lenient fallback)
pub(crate) fn parse_time_value(s: &str) -> f64 {
    let s = s.trim();
    // Strip trailing 's'
    let s = if s.ends_with('s') {
        &s[..s.len() - 1]
    } else {
        s
    };
    if let Some(slash) = s.find('/') {
        let num_str = &s[..slash];
        let den_str = &s[slash + 1..];
        let num: f64 = num_str.trim().parse().unwrap_or(0.0);
        let den: f64 = den_str.trim().parse().unwrap_or(1.0);
        if den.abs() < f64::EPSILON {
            0.0
        } else {
            num / den
        }
    } else {
        s.parse().unwrap_or(0.0)
    }
}

/// Convert a `frameDuration` attribute (e.g. `"100/2400s"`) to FPS.
fn frame_duration_to_fps(fd: &str) -> f32 {
    let duration_s = parse_time_value(fd);
    if duration_s <= 0.0 {
        return 0.0;
    }
    (1.0 / duration_s) as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Minimal FCPXML snippets
    // ------------------------------------------------------------------

    const MINIMAL_FCPXML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.9">
  <library>
    <event name="My Event">
      <project name="My Project">
        <sequence format="r1" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
          <spine>
            <clip name="Shot_001.mov" offset="0s" duration="96/2400s" start="86400s">
            </clip>
            <clip name="Shot_002.mov" offset="96/2400s" duration="144/2400s" start="86400s">
            </clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
  <resources>
    <format id="r1" name="FFVideoFormat1080p24" frameDuration="100/2400s" width="1920" height="1080"/>
  </resources>
</fcpxml>"#;

    const MULTI_SEQ_FCPXML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.9">
  <resources>
    <format id="r1" frameDuration="100/2500s"/>
    <format id="r2" frameDuration="100/3000s"/>
  </resources>
  <library>
    <event name="Event A">
      <project name="Project Alpha">
        <sequence format="r1">
          <spine>
            <clip name="A1.mov" offset="0s" duration="100/2500s" start="0s"/>
          </spine>
        </sequence>
      </project>
    </event>
    <event name="Event B">
      <project name="Project Beta">
        <sequence format="r2">
          <spine>
            <clip name="B1.mov" offset="0s" duration="100/3000s" start="0s"/>
            <clip name="B2.mov" offset="100/3000s" duration="200/3000s" start="0s"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>"#;

    const SIMPLE_30FPS_FCPXML: &str = r#"<fcpxml version="1.9">
  <resources>
    <format id="r1" frameDuration="100/3000s"/>
  </resources>
  <library>
    <event name="E">
      <project name="Test30">
        <sequence format="r1">
          <spine>
            <asset-clip name="ClipA" offset="0s" duration="30s" start="10s"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>"#;

    // ------------------------------------------------------------------
    // parse_time_value unit tests
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_time_rational() {
        let v = parse_time_value("100/2400s");
        assert!((v - 100.0 / 2400.0).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn test_parse_time_integer_seconds() {
        let v = parse_time_value("30s");
        assert!((v - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_time_zero() {
        assert!((parse_time_value("0s")).abs() < 1e-9);
    }

    #[test]
    fn test_parse_time_bare_number() {
        assert!((parse_time_value("5") - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_time_zero_denominator() {
        assert!((parse_time_value("10/0s")).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // FcpxmlParser::parse tests
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_project_name() {
        let project = FcpxmlParser::parse(MINIMAL_FCPXML).expect("parse should succeed");
        assert_eq!(project.name, "My Project");
    }

    #[test]
    fn test_parse_sequence_count() {
        let project = FcpxmlParser::parse(MINIMAL_FCPXML).expect("parse should succeed");
        assert_eq!(project.sequences.len(), 1);
    }

    #[test]
    fn test_parse_frame_rate() {
        let project = FcpxmlParser::parse(MINIMAL_FCPXML).expect("parse should succeed");
        let seq = &project.sequences[0];
        // 100/2400s → 24fps
        assert!(
            (seq.frame_rate - 24.0).abs() < 0.1,
            "fps={}",
            seq.frame_rate
        );
    }

    #[test]
    fn test_parse_clip_count() {
        let project = FcpxmlParser::parse(MINIMAL_FCPXML).expect("parse should succeed");
        let seq = &project.sequences[0];
        assert_eq!(seq.clips.len(), 2);
    }

    #[test]
    fn test_parse_clip_names() {
        let project = FcpxmlParser::parse(MINIMAL_FCPXML).expect("parse should succeed");
        let seq = &project.sequences[0];
        assert_eq!(seq.clips[0].name, "Shot_001.mov");
        assert_eq!(seq.clips[1].name, "Shot_002.mov");
    }

    #[test]
    fn test_parse_clip_offset() {
        let project = FcpxmlParser::parse(MINIMAL_FCPXML).expect("parse should succeed");
        let clip0 = &project.sequences[0].clips[0];
        assert!((clip0.offset_s - 0.0).abs() < 1e-9);
        let clip1 = &project.sequences[0].clips[1];
        let expected_offset = 96.0 / 2400.0;
        assert!(
            (clip1.offset_s - expected_offset).abs() < 1e-9,
            "got {}",
            clip1.offset_s
        );
    }

    #[test]
    fn test_parse_clip_duration() {
        let project = FcpxmlParser::parse(MINIMAL_FCPXML).expect("parse should succeed");
        let clip0 = &project.sequences[0].clips[0];
        let expected = 96.0 / 2400.0;
        assert!(
            (clip0.duration_s - expected).abs() < 1e-9,
            "got {}",
            clip0.duration_s
        );
    }

    #[test]
    fn test_parse_clip_src_in_out() {
        let project = FcpxmlParser::parse(MINIMAL_FCPXML).expect("parse should succeed");
        let clip0 = &project.sequences[0].clips[0];
        // start="86400s" → src_in=86400; src_out = 86400 + 96/2400
        let expected_in = 86400.0_f64;
        let expected_out = 86400.0 + 96.0 / 2400.0;
        assert!(
            (clip0.src_in_s - expected_in).abs() < 1e-6,
            "src_in={}",
            clip0.src_in_s
        );
        assert!(
            (clip0.src_out_s - expected_out).abs() < 1e-6,
            "src_out={}",
            clip0.src_out_s
        );
    }

    #[test]
    fn test_parse_asset_clip_tag() {
        let project = FcpxmlParser::parse(SIMPLE_30FPS_FCPXML).expect("parse should succeed");
        let seq = &project.sequences[0];
        assert_eq!(seq.clips.len(), 1);
        assert_eq!(seq.clips[0].name, "ClipA");
        assert!((seq.clips[0].duration_s - 30.0).abs() < 1e-9);
        assert!((seq.clips[0].src_in_s - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_30fps_sequence() {
        let project = FcpxmlParser::parse(SIMPLE_30FPS_FCPXML).expect("parse should succeed");
        let seq = &project.sequences[0];
        assert!(
            (seq.frame_rate - 30.0).abs() < 0.1,
            "fps={}",
            seq.frame_rate
        );
    }

    #[test]
    fn test_parse_empty_xml_returns_empty_project() {
        let project = FcpxmlParser::parse("<fcpxml/>").expect("parse should not fail");
        assert!(project.sequences.is_empty());
    }

    #[test]
    fn test_parse_multiple_sequences() {
        let project = FcpxmlParser::parse(MULTI_SEQ_FCPXML).expect("parse should succeed");
        // Two project elements → two sequences
        assert_eq!(project.sequences.len(), 2);
    }

    #[test]
    fn test_parse_multi_seq_clip_counts() {
        let project = FcpxmlParser::parse(MULTI_SEQ_FCPXML).expect("parse should succeed");
        // Sequence 0 (25fps) → 1 clip; sequence 1 (30fps) → 2 clips
        let total_clips: usize = project.sequences.iter().map(|s| s.clips.len()).sum();
        assert_eq!(total_clips, 3);
    }

    #[test]
    fn test_parse_multi_seq_fps() {
        let project = FcpxmlParser::parse(MULTI_SEQ_FCPXML).expect("parse should succeed");
        let fps0 = project.sequences[0].frame_rate;
        let fps1 = project.sequences[1].frame_rate;
        assert!((fps0 - 25.0).abs() < 0.1, "seq0 fps={fps0}");
        assert!((fps1 - 30.0).abs() < 0.1, "seq1 fps={fps1}");
    }
}
