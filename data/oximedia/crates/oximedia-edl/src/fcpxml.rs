//! XML-based EDL format support (Final Cut Pro XML style).
//!
//! This module provides parsing and generation of simplified FCP XML EDL
//! format, focusing on the clip/sequence timeline representation commonly
//! exported from Final Cut Pro, DaVinci Resolve, and similar NLEs.
//!
//! The format represents sequences as XML trees with clip references,
//! timecodes, and transition information.

#![allow(dead_code)]

use crate::error::{EdlError, EdlResult};
use std::fmt::Write as FmtWrite;

/// Represents an FCP XML sequence (simplified).
#[derive(Debug, Clone)]
pub struct FcpXmlSequence {
    /// Sequence name/title.
    pub name: String,
    /// Sequence UUID.
    pub uuid: String,
    /// Duration in frames.
    pub duration_frames: u64,
    /// Timebase (frames per second, integer nominal).
    pub timebase: u32,
    /// Whether timecodes use NTSC drop-frame.
    pub ntsc: bool,
    /// Clips on the timeline.
    pub clips: Vec<FcpXmlClip>,
    /// Transitions between clips.
    pub transitions: Vec<FcpXmlTransition>,
}

/// A single clip in an FCP XML sequence.
#[derive(Debug, Clone)]
pub struct FcpXmlClip {
    /// Clip name.
    pub name: String,
    /// Source file ID or path.
    pub file_ref: String,
    /// Source in point (frames).
    pub source_in: u64,
    /// Source out point (frames).
    pub source_out: u64,
    /// Timeline in point (frames).
    pub timeline_in: u64,
    /// Timeline out point (frames).
    pub timeline_out: u64,
    /// Whether this is a video clip.
    pub has_video: bool,
    /// Whether this is an audio clip.
    pub has_audio: bool,
    /// Optional reel/tape name.
    pub reel: Option<String>,
}

impl FcpXmlClip {
    /// Create a new clip.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        file_ref: impl Into<String>,
        source_in: u64,
        source_out: u64,
        timeline_in: u64,
        timeline_out: u64,
    ) -> Self {
        Self {
            name: name.into(),
            file_ref: file_ref.into(),
            source_in,
            source_out,
            timeline_in,
            timeline_out,
            has_video: true,
            has_audio: true,
            reel: None,
        }
    }

    /// Duration in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.timeline_out.saturating_sub(self.timeline_in)
    }

    /// Source duration in frames.
    #[must_use]
    pub fn source_duration_frames(&self) -> u64 {
        self.source_out.saturating_sub(self.source_in)
    }
}

/// A transition between clips.
#[derive(Debug, Clone)]
pub struct FcpXmlTransition {
    /// Transition name/type (e.g., "Cross Dissolve").
    pub name: String,
    /// Duration in frames.
    pub duration_frames: u64,
    /// Start position on timeline (frames).
    pub timeline_start: u64,
}

impl FcpXmlSequence {
    /// Create a new empty sequence.
    #[must_use]
    pub fn new(name: impl Into<String>, timebase: u32) -> Self {
        Self {
            name: name.into(),
            uuid: String::new(),
            duration_frames: 0,
            timebase,
            ntsc: false,
            clips: Vec::new(),
            transitions: Vec::new(),
        }
    }

    /// Add a clip to the sequence.
    pub fn add_clip(&mut self, clip: FcpXmlClip) {
        if clip.timeline_out > self.duration_frames {
            self.duration_frames = clip.timeline_out;
        }
        self.clips.push(clip);
    }

    /// Add a transition.
    pub fn add_transition(&mut self, transition: FcpXmlTransition) {
        self.transitions.push(transition);
    }

    /// Get clip count.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Get total duration in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        if self.timebase == 0 {
            return 0.0;
        }
        self.duration_frames as f64 / self.timebase as f64
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Simple XML parser (no external XML crate — pure Rust)
// ────────────────────────────────────────────────────────────────────────────

/// Parse a simplified FCP XML string into an `FcpXmlSequence`.
///
/// This parser handles the core structure of FCP XML exports.
/// It is intentionally lenient and focuses on extracting clip and
/// timeline data rather than full XML compliance.
///
/// # Errors
///
/// Returns an error if the XML structure is invalid or missing required elements.
pub fn parse_fcpxml(input: &str) -> EdlResult<FcpXmlSequence> {
    // Extract sequence name
    let name = extract_tag_content(input, "name").unwrap_or_else(|| "Untitled".to_string());

    // Extract timebase
    let timebase_str = extract_tag_content(input, "timebase").unwrap_or_else(|| "25".to_string());
    let timebase = timebase_str
        .trim()
        .parse::<u32>()
        .map_err(|_| EdlError::parse(0, format!("Invalid timebase: {timebase_str}")))?;

    // Extract NTSC flag
    let ntsc =
        extract_tag_content(input, "ntsc").is_some_and(|v| v.trim().eq_ignore_ascii_case("TRUE"));

    // Extract duration
    let duration_str = extract_tag_content(input, "duration").unwrap_or_else(|| "0".to_string());
    let duration_frames = duration_str.trim().parse::<u64>().unwrap_or(0);

    let mut seq = FcpXmlSequence::new(name, timebase);
    seq.ntsc = ntsc;
    seq.duration_frames = duration_frames;

    // Extract clips from <clipitem> elements
    let mut search_pos = 0;
    while let Some(clip_start) = input[search_pos..].find("<clipitem") {
        let abs_start = search_pos + clip_start;
        if let Some(clip_end) = input[abs_start..].find("</clipitem>") {
            let clip_xml = &input[abs_start..abs_start + clip_end + "</clipitem>".len()];
            if let Some(clip) = parse_clipitem(clip_xml) {
                seq.clips.push(clip);
            }
            search_pos = abs_start + clip_end + "</clipitem>".len();
        } else {
            break;
        }
    }

    // Extract transitions from <transitionitem> elements
    search_pos = 0;
    while let Some(tr_start) = input[search_pos..].find("<transitionitem") {
        let abs_start = search_pos + tr_start;
        if let Some(tr_end) = input[abs_start..].find("</transitionitem>") {
            let tr_xml = &input[abs_start..abs_start + tr_end + "</transitionitem>".len()];
            if let Some(transition) = parse_transitionitem(tr_xml) {
                seq.transitions.push(transition);
            }
            search_pos = abs_start + tr_end + "</transitionitem>".len();
        } else {
            break;
        }
    }

    Ok(seq)
}

/// Extract content between opening and closing tags of given name.
fn extract_tag_content(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)?;
    let content_start = start + open.len();
    let end = xml[content_start..].find(&close)?;
    Some(xml[content_start..content_start + end].to_string())
}

/// Parse a single `<clipitem>` element.
fn parse_clipitem(xml: &str) -> Option<FcpXmlClip> {
    let name = extract_tag_content(xml, "name").unwrap_or_default();

    let file_ref = extract_tag_content(xml, "file")
        .or_else(|| extract_attribute(xml, "clipitem", "id"))
        .unwrap_or_default();

    let src_in = extract_tag_content(xml, "in")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let src_out = extract_tag_content(xml, "out")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let tl_in = extract_tag_content(xml, "start")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(src_in);
    let tl_out = extract_tag_content(xml, "end")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(src_out);

    let reel = extract_tag_content(xml, "reel");

    let mut clip = FcpXmlClip::new(name, file_ref, src_in, src_out, tl_in, tl_out);
    clip.reel = reel;
    Some(clip)
}

/// Parse a `<transitionitem>` element.
fn parse_transitionitem(xml: &str) -> Option<FcpXmlTransition> {
    let name = extract_tag_content(xml, "name").unwrap_or_else(|| "Unknown".to_string());
    let duration = extract_tag_content(xml, "duration")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let start = extract_tag_content(xml, "start")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);

    Some(FcpXmlTransition {
        name,
        duration_frames: duration,
        timeline_start: start,
    })
}

/// Extract an attribute value from an XML opening tag.
fn extract_attribute(xml: &str, tag: &str, attr: &str) -> Option<String> {
    let tag_start = xml.find(&format!("<{tag}"))?;
    let tag_end = xml[tag_start..].find('>')?;
    let tag_text = &xml[tag_start..tag_start + tag_end];
    let attr_pattern = format!("{attr}=\"");
    let attr_start = tag_text.find(&attr_pattern)?;
    let value_start = attr_start + attr_pattern.len();
    let value_end = tag_text[value_start..].find('"')?;
    Some(tag_text[value_start..value_start + value_end].to_string())
}

// ────────────────────────────────────────────────────────────────────────────
// Generator
// ────────────────────────────────────────────────────────────────────────────

/// Generate an FCP XML string from a sequence.
///
/// # Errors
///
/// Returns an error if generation fails.
pub fn generate_fcpxml(seq: &FcpXmlSequence) -> EdlResult<String> {
    let mut output = String::new();

    writeln!(output, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "<!DOCTYPE xmeml>")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "<xmeml version=\"5\">")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "  <sequence>")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "    <name>{}</name>", seq.name)
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "    <duration>{}</duration>", seq.duration_frames)
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "    <rate>")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "      <timebase>{}</timebase>", seq.timebase)
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(
        output,
        "      <ntsc>{}</ntsc>",
        if seq.ntsc { "TRUE" } else { "FALSE" }
    )
    .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "    </rate>")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;

    // Write clips
    writeln!(output, "    <media>")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "      <video>")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "        <track>")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;

    for clip in &seq.clips {
        write_clipitem(&mut output, clip)?;
    }

    for transition in &seq.transitions {
        write_transitionitem(&mut output, transition)?;
    }

    writeln!(output, "        </track>")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "      </video>")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "    </media>")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "  </sequence>")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
    writeln!(output, "</xmeml>")
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;

    Ok(output)
}

fn write_clipitem(output: &mut String, clip: &FcpXmlClip) -> EdlResult<()> {
    let map_err = |e: std::fmt::Error| EdlError::ValidationError(format!("Write error: {e}"));
    writeln!(output, "          <clipitem>").map_err(map_err)?;
    writeln!(output, "            <name>{}</name>", clip.name).map_err(map_err)?;
    writeln!(output, "            <file>{}</file>", clip.file_ref).map_err(map_err)?;
    writeln!(output, "            <in>{}</in>", clip.source_in).map_err(map_err)?;
    writeln!(output, "            <out>{}</out>", clip.source_out).map_err(map_err)?;
    writeln!(output, "            <start>{}</start>", clip.timeline_in).map_err(map_err)?;
    writeln!(output, "            <end>{}</end>", clip.timeline_out).map_err(map_err)?;
    if let Some(reel) = &clip.reel {
        writeln!(output, "            <reel>{reel}</reel>").map_err(map_err)?;
    }
    writeln!(output, "          </clipitem>").map_err(map_err)?;
    Ok(())
}

fn write_transitionitem(output: &mut String, tr: &FcpXmlTransition) -> EdlResult<()> {
    let map_err = |e: std::fmt::Error| EdlError::ValidationError(format!("Write error: {e}"));
    writeln!(output, "          <transitionitem>").map_err(map_err)?;
    writeln!(output, "            <name>{}</name>", tr.name).map_err(map_err)?;
    writeln!(
        output,
        "            <duration>{}</duration>",
        tr.duration_frames
    )
    .map_err(map_err)?;
    writeln!(output, "            <start>{}</start>", tr.timeline_start).map_err(map_err)?;
    writeln!(output, "          </transitionitem>").map_err(map_err)?;
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="5">
  <sequence>
    <name>Test Sequence</name>
    <duration>250</duration>
    <rate>
      <timebase>25</timebase>
      <ntsc>FALSE</ntsc>
    </rate>
    <media>
      <video>
        <track>
          <clipitem id="clip1">
            <name>Shot 1</name>
            <file>file1.mov</file>
            <in>0</in>
            <out>125</out>
            <start>0</start>
            <end>125</end>
            <reel>A001</reel>
          </clipitem>
          <clipitem id="clip2">
            <name>Shot 2</name>
            <file>file2.mov</file>
            <in>0</in>
            <out>125</out>
            <start>125</start>
            <end>250</end>
          </clipitem>
          <transitionitem>
            <name>Cross Dissolve</name>
            <duration>15</duration>
            <start>118</start>
          </transitionitem>
        </track>
      </video>
    </media>
  </sequence>
</xmeml>"#;

    #[test]
    fn test_parse_fcpxml_sequence() {
        let seq = parse_fcpxml(SAMPLE_XML).expect("parse should succeed");
        assert_eq!(seq.name, "Test Sequence");
        assert_eq!(seq.timebase, 25);
        assert!(!seq.ntsc);
        assert_eq!(seq.duration_frames, 250);
    }

    #[test]
    fn test_parse_fcpxml_clips() {
        let seq = parse_fcpxml(SAMPLE_XML).expect("parse should succeed");
        assert_eq!(seq.clip_count(), 2);

        assert_eq!(seq.clips[0].name, "Shot 1");
        assert_eq!(seq.clips[0].source_in, 0);
        assert_eq!(seq.clips[0].source_out, 125);
        assert_eq!(seq.clips[0].timeline_in, 0);
        assert_eq!(seq.clips[0].timeline_out, 125);
        assert_eq!(seq.clips[0].reel, Some("A001".to_string()));
    }

    #[test]
    fn test_parse_fcpxml_second_clip() {
        let seq = parse_fcpxml(SAMPLE_XML).expect("parse should succeed");
        assert_eq!(seq.clips[1].name, "Shot 2");
        assert_eq!(seq.clips[1].timeline_in, 125);
        assert_eq!(seq.clips[1].timeline_out, 250);
        assert_eq!(seq.clips[1].reel, None);
    }

    #[test]
    fn test_parse_fcpxml_transitions() {
        let seq = parse_fcpxml(SAMPLE_XML).expect("parse should succeed");
        assert_eq!(seq.transitions.len(), 1);
        assert_eq!(seq.transitions[0].name, "Cross Dissolve");
        assert_eq!(seq.transitions[0].duration_frames, 15);
        assert_eq!(seq.transitions[0].timeline_start, 118);
    }

    #[test]
    fn test_clip_duration() {
        let clip = FcpXmlClip::new("test", "file", 10, 60, 100, 150);
        assert_eq!(clip.duration_frames(), 50);
        assert_eq!(clip.source_duration_frames(), 50);
    }

    #[test]
    fn test_sequence_duration_seconds() {
        let mut seq = FcpXmlSequence::new("test", 25);
        seq.duration_frames = 250;
        assert!((seq.duration_seconds() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_sequence_duration_seconds_zero_timebase() {
        let seq = FcpXmlSequence::new("test", 0);
        assert_eq!(seq.duration_seconds(), 0.0);
    }

    #[test]
    fn test_generate_fcpxml() {
        let mut seq = FcpXmlSequence::new("My Seq", 25);
        seq.duration_frames = 100;
        seq.add_clip(FcpXmlClip::new("Clip1", "file1.mov", 0, 50, 0, 50));
        seq.add_clip(FcpXmlClip::new("Clip2", "file2.mov", 0, 50, 50, 100));

        let xml = generate_fcpxml(&seq).expect("generate should succeed");
        assert!(xml.contains("<name>My Seq</name>"));
        assert!(xml.contains("<timebase>25</timebase>"));
        assert!(xml.contains("<name>Clip1</name>"));
        assert!(xml.contains("<name>Clip2</name>"));
    }

    #[test]
    fn test_generate_fcpxml_roundtrip() {
        let seq = parse_fcpxml(SAMPLE_XML).expect("parse should succeed");
        let generated = generate_fcpxml(&seq).expect("generate should succeed");
        let reparsed = parse_fcpxml(&generated).expect("reparse should succeed");

        assert_eq!(reparsed.name, seq.name);
        assert_eq!(reparsed.timebase, seq.timebase);
        assert_eq!(reparsed.clip_count(), seq.clip_count());

        for (orig, re) in seq.clips.iter().zip(reparsed.clips.iter()) {
            assert_eq!(orig.name, re.name);
            assert_eq!(orig.source_in, re.source_in);
            assert_eq!(orig.source_out, re.source_out);
            assert_eq!(orig.timeline_in, re.timeline_in);
            assert_eq!(orig.timeline_out, re.timeline_out);
        }
    }

    #[test]
    fn test_generate_fcpxml_with_transition() {
        let mut seq = FcpXmlSequence::new("Tr Test", 30);
        seq.add_transition(FcpXmlTransition {
            name: "Dissolve".to_string(),
            duration_frames: 20,
            timeline_start: 80,
        });

        let xml = generate_fcpxml(&seq).expect("generate should succeed");
        assert!(xml.contains("<transitionitem>"));
        assert!(xml.contains("<name>Dissolve</name>"));
        assert!(xml.contains("<duration>20</duration>"));
    }

    #[test]
    fn test_add_clip_updates_duration() {
        let mut seq = FcpXmlSequence::new("test", 25);
        assert_eq!(seq.duration_frames, 0);
        seq.add_clip(FcpXmlClip::new("c1", "f1", 0, 100, 0, 100));
        assert_eq!(seq.duration_frames, 100);
        seq.add_clip(FcpXmlClip::new("c2", "f2", 0, 50, 100, 150));
        assert_eq!(seq.duration_frames, 150);
    }

    #[test]
    fn test_ntsc_flag() {
        let xml = r#"<sequence><name>NTSC</name><duration>100</duration>
        <rate><timebase>30</timebase><ntsc>TRUE</ntsc></rate></sequence>"#;
        let seq = parse_fcpxml(xml).expect("parse should succeed");
        assert!(seq.ntsc);
        assert_eq!(seq.timebase, 30);
    }

    #[test]
    fn test_extract_attribute() {
        let xml = r#"<clipitem id="abc123"><name>test</name></clipitem>"#;
        let attr = extract_attribute(xml, "clipitem", "id");
        assert_eq!(attr, Some("abc123".to_string()));
    }
}
