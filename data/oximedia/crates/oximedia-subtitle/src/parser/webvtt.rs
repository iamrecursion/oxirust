//! WebVTT (Web Video Text Tracks) subtitle parser.
//!
//! WebVTT is a W3C standard format for web video captions.
//! Supports region definitions, vertical text cue settings, and positioning.
//!
//! ```text
//! WEBVTT
//!
//! REGION
//! id:bottom
//! width:40%
//! lines:3
//! regionanchor:0%,100%
//! viewportanchor:10%,90%
//! scroll:up
//!
//! 00:00:01.000 --> 00:00:04.000
//! This is the first subtitle.
//!
//! 00:00:05.000 --> 00:00:08.000 position:50% align:middle vertical:rl
//! This is a positioned vertical subtitle.
//! ```

use crate::style::{Alignment, Position};
use crate::{Subtitle, SubtitleError, SubtitleResult};
use std::collections::HashMap;

// ============================================================================
// Region and vertical text types
// ============================================================================

/// Vertical text writing direction for WebVTT cues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerticalText {
    /// Horizontal text (default).
    Horizontal,
    /// Right-to-left vertical writing (columns flow right-to-left).
    Rl,
    /// Left-to-right vertical writing (columns flow left-to-right).
    Lr,
}

impl Default for VerticalText {
    fn default() -> Self {
        Self::Horizontal
    }
}

/// Scroll direction for a WebVTT region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionScroll {
    /// No scrolling (static region).
    None,
    /// Lines scroll upward as new lines are added.
    Up,
}

/// A WebVTT region definition (RFC 8216 / WebVTT spec).
#[derive(Clone, Debug)]
pub struct VttRegion {
    /// Region identifier.
    pub id: String,
    /// Width as a percentage of the viewport (0.0–100.0).
    pub width_pct: f32,
    /// Number of lines visible in the region.
    pub lines: u32,
    /// Region anchor X as a percentage.
    pub region_anchor_x: f32,
    /// Region anchor Y as a percentage.
    pub region_anchor_y: f32,
    /// Viewport anchor X as a percentage.
    pub viewport_anchor_x: f32,
    /// Viewport anchor Y as a percentage.
    pub viewport_anchor_y: f32,
    /// Scroll behaviour.
    pub scroll: RegionScroll,
}

impl Default for VttRegion {
    fn default() -> Self {
        Self {
            id: String::new(),
            width_pct: 100.0,
            lines: 3,
            region_anchor_x: 0.0,
            region_anchor_y: 100.0,
            viewport_anchor_x: 0.0,
            viewport_anchor_y: 100.0,
            scroll: RegionScroll::None,
        }
    }
}

/// Extended cue settings extracted from a WebVTT cue timing line.
#[derive(Clone, Debug, Default)]
pub struct CueSettings {
    /// Horizontal position percentage (0–100).
    pub position_pct: Option<f32>,
    /// Percentage size of the cue box.
    pub size_pct: Option<f32>,
    /// Vertical line position (negative = from bottom).
    pub line_pct: Option<f32>,
    /// Text alignment within the cue.
    pub alignment: Option<Alignment>,
    /// Vertical writing direction.
    pub vertical: VerticalText,
    /// Region identifier this cue is anchored to.
    pub region: Option<String>,
}

/// Parse a percent value like "50%" → 50.0.
fn parse_percent(s: &str) -> Option<f32> {
    s.trim_end_matches('%').parse::<f32>().ok()
}

/// Parse cue settings from a settings string.
fn parse_cue_settings_extended(settings: &str) -> CueSettings {
    let mut cs = CueSettings::default();

    for token in settings.split_whitespace() {
        if let Some(val) = token.strip_prefix("position:") {
            cs.position_pct = parse_percent(val);
        } else if let Some(val) = token.strip_prefix("size:") {
            cs.size_pct = parse_percent(val);
        } else if let Some(val) = token.strip_prefix("line:") {
            // line can be "N%" or "-N%" or integer snap-to-lines
            cs.line_pct = parse_percent(val).or_else(|| val.parse::<f32>().ok());
        } else if let Some(val) = token.strip_prefix("align:") {
            cs.alignment = match val {
                "start" | "left" => Some(Alignment::Left),
                "center" | "middle" => Some(Alignment::Center),
                "end" | "right" => Some(Alignment::Right),
                _ => None,
            };
        } else if let Some(val) = token.strip_prefix("vertical:") {
            cs.vertical = match val {
                "rl" => VerticalText::Rl,
                "lr" => VerticalText::Lr,
                _ => VerticalText::Horizontal,
            };
        } else if let Some(val) = token.strip_prefix("region:") {
            cs.region = Some(val.to_string());
        }
    }

    cs
}

/// Parse a REGION block from a WebVTT file.
///
/// Input should start at the line after "REGION".
fn parse_region_block(lines: &[&str]) -> VttRegion {
    let mut region = VttRegion::default();

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        if let Some(val) = line.strip_prefix("id:") {
            region.id = val.trim().to_string();
        } else if let Some(val) = line.strip_prefix("width:") {
            if let Some(pct) = parse_percent(val.trim()) {
                region.width_pct = pct;
            }
        } else if let Some(val) = line.strip_prefix("lines:") {
            if let Ok(n) = val.trim().parse::<u32>() {
                region.lines = n;
            }
        } else if let Some(val) = line.strip_prefix("regionanchor:") {
            let parts: Vec<&str> = val.split(',').collect();
            if parts.len() == 2 {
                if let Some(x) = parse_percent(parts[0].trim()) {
                    region.region_anchor_x = x;
                }
                if let Some(y) = parse_percent(parts[1].trim()) {
                    region.region_anchor_y = y;
                }
            }
        } else if let Some(val) = line.strip_prefix("viewportanchor:") {
            let parts: Vec<&str> = val.split(',').collect();
            if parts.len() == 2 {
                if let Some(x) = parse_percent(parts[0].trim()) {
                    region.viewport_anchor_x = x;
                }
                if let Some(y) = parse_percent(parts[1].trim()) {
                    region.viewport_anchor_y = y;
                }
            }
        } else if let Some(val) = line.strip_prefix("scroll:") {
            region.scroll = match val.trim() {
                "up" => RegionScroll::Up,
                _ => RegionScroll::None,
            };
        }
    }

    region
}

/// A fully parsed WebVTT document including regions.
#[derive(Clone, Debug, Default)]
pub struct VttDocument {
    /// Named regions defined in the file.
    pub regions: HashMap<String, VttRegion>,
    /// Subtitle cues.
    pub cues: Vec<Subtitle>,
    /// Extended cue settings (parallel to `cues`).
    pub cue_settings: Vec<CueSettings>,
}

/// Parse a WebVTT document including region definitions.
///
/// # Errors
///
/// Returns error if parsing fails.
pub fn parse_document(input: &str) -> SubtitleResult<VttDocument> {
    let normalized = input.replace("\r\n", "\n");
    let lines: Vec<&str> = normalized.lines().collect();

    // Verify header
    if lines.is_empty()
        || !lines[0]
            .trim_start_matches('\u{feff}')
            .starts_with("WEBVTT")
    {
        return Err(SubtitleError::ParseError(
            "Missing WEBVTT header".to_string(),
        ));
    }

    let mut doc = VttDocument::default();
    let mut i = 1usize;

    // Skip rest of header line
    while i < lines.len() && !lines[i].is_empty() {
        i += 1;
    }
    i += 1; // skip blank line after header

    while i < lines.len() {
        let line = lines[i].trim();

        if line.is_empty() {
            i += 1;
            continue;
        }

        // REGION block
        if line == "REGION" {
            i += 1;
            let start = i;
            // Collect region property lines
            while i < lines.len() && !lines[i].trim().is_empty() {
                i += 1;
            }
            let region = parse_region_block(&lines[start..i]);
            if !region.id.is_empty() {
                doc.regions.insert(region.id.clone(), region);
            }
            continue;
        }

        // NOTE block — skip
        if line.starts_with("NOTE") {
            i += 1;
            while i < lines.len() && !lines[i].trim().is_empty() {
                i += 1;
            }
            continue;
        }

        // STYLE block — skip
        if line.starts_with("STYLE") {
            i += 1;
            while i < lines.len() && !lines[i].trim().is_empty() {
                i += 1;
            }
            continue;
        }

        // Cue identifier (optional line before timing line)
        let mut cue_id: Option<String> = None;
        let timing_line;
        if line.contains("-->") {
            timing_line = line;
        } else {
            // This is a cue identifier
            cue_id = Some(line.to_string());
            i += 1;
            if i >= lines.len() {
                break;
            }
            timing_line = lines[i].trim();
        }

        // Parse timing line
        if !timing_line.contains("-->") {
            i += 1;
            continue;
        }

        let (start_ms, end_ms, settings_str) = match parse_timing_str(timing_line) {
            Some(t) => t,
            None => {
                i += 1;
                continue;
            }
        };

        let cue_settings = parse_cue_settings_extended(settings_str);
        i += 1;

        // Collect payload lines
        let mut text_lines: Vec<&str> = Vec::new();
        while i < lines.len() && !lines[i].trim().is_empty() {
            text_lines.push(lines[i]);
            i += 1;
        }

        let raw_text = text_lines.join("\n");
        let cleaned_text = strip_vtt_tags(&raw_text);

        // Determine position from cue settings
        let position = cue_settings
            .position_pct
            .map(|pct| Position::new(pct / 100.0, 0.9));

        let mut sub = Subtitle::new(start_ms, end_ms, cleaned_text);
        if let Some(id) = cue_id {
            sub = sub.with_id(id);
        }
        if let Some(pos) = position {
            sub.position = Some(pos);
        }

        doc.cues.push(sub);
        doc.cue_settings.push(cue_settings);
    }

    Ok(doc)
}

/// Parse a timing string like "00:00:01.000 --> 00:00:04.000 align:start".
/// Returns `(start_ms, end_ms, settings_str)` or `None`.
fn parse_timing_str(line: &str) -> Option<(i64, i64, &str)> {
    let arrow = line.find("-->")?;
    let start_str = line[..arrow].trim();
    let rest = line[arrow + 3..].trim();

    // Split rest into end_time and optional settings
    let (end_str, settings_str) = if let Some(sp) = rest.find(|c: char| c.is_whitespace()) {
        (&rest[..sp], &rest[sp + 1..])
    } else {
        (rest, "")
    };

    let start_ms = parse_vtt_timestamp_str(start_str)?;
    let end_ms = parse_vtt_timestamp_str(end_str)?;
    Some((start_ms, end_ms, settings_str))
}

/// Parse a VTT timestamp string (HH:MM:SS.mmm or MM:SS.mmm).
fn parse_vtt_timestamp_str(s: &str) -> Option<i64> {
    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        3 => {
            let h: i64 = parts[0].parse().ok()?;
            let m: i64 = parts[1].parse().ok()?;
            let (sec_s, ms_s) = parts[2].split_once('.')?;
            let sec: i64 = sec_s.parse().ok()?;
            let ms: i64 = ms_s.parse().ok()?;
            Some(h * 3_600_000 + m * 60_000 + sec * 1_000 + ms)
        }
        2 => {
            let m: i64 = parts[0].parse().ok()?;
            let (sec_s, ms_s) = parts[1].split_once('.')?;
            let sec: i64 = sec_s.parse().ok()?;
            let ms: i64 = ms_s.parse().ok()?;
            Some(m * 60_000 + sec * 1_000 + ms)
        }
        _ => None,
    }
}

// ============================================================================
// Legacy nom-based parser (kept for backward compatibility)
// ============================================================================

/// Parse WebVTT subtitle file.
///
/// # Errors
///
/// Returns error if the file is not valid WebVTT format.
pub fn parse(data: &[u8]) -> SubtitleResult<Vec<Subtitle>> {
    let text = String::from_utf8_lossy(data);
    parse_webvtt(&text)
}

/// Parse WebVTT subtitle from string.
///
/// # Errors
///
/// Returns error if parsing fails.
pub fn parse_webvtt(input: &str) -> SubtitleResult<Vec<Subtitle>> {
    // Use the new document parser which handles regions and vertical text properly.
    let doc = parse_document(input)?;
    Ok(doc.cues)
}

/// Strip WebVTT formatting tags.
fn strip_vtt_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_tag = false;

    for c in text.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ => {
                if !in_tag {
                    result.push(c);
                }
            }
        }
    }

    crate::text::decode_html_entities(&result)
}

/// Format milliseconds as WebVTT timestamp.
#[must_use]
pub fn format_timestamp(ms: i64) -> String {
    let hours = ms / 3600000;
    let minutes = (ms % 3600000) / 60000;
    let seconds = (ms % 60000) / 1000;
    let millis = ms % 1000;

    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
    } else {
        format!("{minutes:02}:{seconds:02}.{millis:03}")
    }
}

/// Write subtitles in WebVTT format.
///
/// # Errors
///
/// Returns error if writing fails.
pub fn write(subtitles: &[Subtitle]) -> SubtitleResult<String> {
    let mut output = String::from("WEBVTT\n\n");

    for subtitle in subtitles {
        // Timestamps
        output.push_str(&format!(
            "{} --> {}\n",
            format_timestamp(subtitle.start_time),
            format_timestamp(subtitle.end_time)
        ));

        // Text
        output.push_str(&subtitle.text);
        output.push_str("\n\n");
    }

    Ok(output)
}

/// Write subtitles in WebVTT format including region definitions.
///
/// # Errors
///
/// Returns error if writing fails.
pub fn write_document(doc: &VttDocument) -> SubtitleResult<String> {
    let mut output = String::from("WEBVTT\n\n");

    // Write regions
    for (_, region) in &doc.regions {
        output.push_str("REGION\n");
        output.push_str(&format!("id:{}\n", region.id));
        output.push_str(&format!("width:{:.0}%\n", region.width_pct));
        output.push_str(&format!("lines:{}\n", region.lines));
        output.push_str(&format!(
            "regionanchor:{:.0}%,{:.0}%\n",
            region.region_anchor_x, region.region_anchor_y
        ));
        output.push_str(&format!(
            "viewportanchor:{:.0}%,{:.0}%\n",
            region.viewport_anchor_x, region.viewport_anchor_y
        ));
        if region.scroll == RegionScroll::Up {
            output.push_str("scroll:up\n");
        }
        output.push('\n');
    }

    for (sub, settings) in doc.cues.iter().zip(doc.cue_settings.iter()) {
        let mut timing = format!(
            "{} --> {}",
            format_timestamp(sub.start_time),
            format_timestamp(sub.end_time)
        );

        // Append settings that are set
        if let Some(pct) = settings.position_pct {
            timing.push_str(&format!(" position:{pct:.0}%"));
        }
        if let Some(pct) = settings.size_pct {
            timing.push_str(&format!(" size:{pct:.0}%"));
        }
        match settings.vertical {
            VerticalText::Rl => timing.push_str(" vertical:rl"),
            VerticalText::Lr => timing.push_str(" vertical:lr"),
            VerticalText::Horizontal => {}
        }
        if let Some(align) = &settings.alignment {
            let a = match align {
                Alignment::Left => "start",
                Alignment::Center => "center",
                Alignment::Right => "end",
            };
            timing.push_str(&format!(" align:{a}"));
        }
        if let Some(region) = &settings.region {
            timing.push_str(&format!(" region:{region}"));
        }

        output.push_str(&timing);
        output.push('\n');
        output.push_str(&sub.text);
        output.push_str("\n\n");
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASIC_VTT: &str = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nHello VTT!\n\n00:00:05.000 --> 00:00:09.000\nSecond cue.\n\n";

    #[test]
    fn test_parse_basic_webvtt() {
        let subs = parse_webvtt(BASIC_VTT).expect("parse_webvtt basic");
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].text, "Hello VTT!");
        assert_eq!(subs[0].start_time, 1000);
        assert_eq!(subs[0].end_time, 4000);
    }

    #[test]
    fn test_parse_cue_identifier() {
        let vtt = "WEBVTT\n\ncue1\n00:00:01.000 --> 00:00:04.000\nWith ID.\n\n";
        let subs = parse_webvtt(vtt).expect("parse_webvtt with id");
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].id, Some("cue1".to_string()));
    }

    #[test]
    fn test_parse_position_setting() {
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000 position:75%\nPositioned.\n\n";
        let subs = parse_webvtt(vtt).expect("parse position");
        assert!(subs[0].position.is_some());
        let pos = subs[0].position.as_ref().expect("position set");
        assert!((pos.x - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_parse_region_definition() {
        let vtt = "WEBVTT\n\nREGION\nid:bottom\nwidth:40%\nlines:3\nregionanchor:0%,100%\nviewportanchor:10%,90%\nscroll:up\n\n00:00:01.000 --> 00:00:04.000\nHi\n\n";
        let doc = parse_document(vtt).expect("parse region doc");
        assert!(doc.regions.contains_key("bottom"));
        let region = &doc.regions["bottom"];
        assert!((region.width_pct - 40.0).abs() < 0.01);
        assert_eq!(region.lines, 3);
        assert_eq!(region.scroll, RegionScroll::Up);
        assert!((region.region_anchor_y - 100.0).abs() < 0.01);
        assert!((region.viewport_anchor_x - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_vertical_text_rl() {
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000 vertical:rl\nVertical.\n\n";
        let doc = parse_document(vtt).expect("parse vertical doc");
        assert_eq!(doc.cue_settings[0].vertical, VerticalText::Rl);
    }

    #[test]
    fn test_parse_vertical_text_lr() {
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000 vertical:lr\nVertical LR.\n\n";
        let doc = parse_document(vtt).expect("parse vertical lr doc");
        assert_eq!(doc.cue_settings[0].vertical, VerticalText::Lr);
    }

    #[test]
    fn test_parse_multiple_cue_settings() {
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000 position:50% size:80% align:center vertical:rl\nTest.\n\n";
        let doc = parse_document(vtt).expect("parse multi settings");
        let cs = &doc.cue_settings[0];
        assert!(cs.position_pct.is_some());
        assert!((cs.position_pct.expect("position") - 50.0).abs() < 0.01);
        assert!(cs.size_pct.is_some());
        assert!((cs.size_pct.expect("size") - 80.0).abs() < 0.01);
        assert_eq!(cs.alignment, Some(Alignment::Center));
        assert_eq!(cs.vertical, VerticalText::Rl);
    }

    #[test]
    fn test_parse_region_anchor_in_cue() {
        let vtt = "WEBVTT\n\nREGION\nid:top\nwidth:60%\nlines:2\n\n00:00:01.000 --> 00:00:04.000 region:top\nIn region.\n\n";
        let doc = parse_document(vtt).expect("parse region cue");
        assert_eq!(doc.cue_settings[0].region, Some("top".to_string()));
    }

    #[test]
    fn test_skip_note_block() {
        let vtt = "WEBVTT\n\nNOTE This is a comment\nspanning two lines\n\n00:00:01.000 --> 00:00:04.000\nHello.\n\n";
        let subs = parse_webvtt(vtt).expect("parse with NOTE");
        assert_eq!(subs.len(), 1);
    }

    #[test]
    fn test_strip_vtt_tags() {
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<b>Bold</b> and <i>italic</i>.\n\n";
        let subs = parse_webvtt(vtt).expect("parse with tags");
        assert_eq!(subs[0].text, "Bold and italic.");
    }

    #[test]
    fn test_parse_bom_prefix() {
        let vtt = "\u{feff}WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nBOM test.\n\n";
        let subs = parse_webvtt(vtt).expect("parse with BOM");
        assert_eq!(subs.len(), 1);
    }

    #[test]
    fn test_format_timestamp_hms() {
        assert_eq!(format_timestamp(3_661_000), "01:01:01.000");
    }

    #[test]
    fn test_format_timestamp_ms_only() {
        assert_eq!(format_timestamp(61_500), "01:01.500");
    }

    #[test]
    fn test_write_roundtrip() {
        let subs = vec![
            Subtitle::new(1000, 4000, "Hello".to_string()),
            Subtitle::new(5000, 8000, "World".to_string()),
        ];
        let vtt = write(&subs).expect("write vtt");
        assert!(vtt.starts_with("WEBVTT"));
        let parsed = parse_webvtt(&vtt).expect("roundtrip parse");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].start_time, 1000);
        assert_eq!(parsed[1].text, "World");
    }

    #[test]
    fn test_write_document_with_region() {
        let mut doc = VttDocument::default();
        let mut region = VttRegion::default();
        region.id = "r1".to_string();
        region.width_pct = 50.0;
        region.lines = 2;
        region.scroll = RegionScroll::Up;
        doc.regions.insert("r1".to_string(), region);

        let sub = Subtitle::new(1000, 3000, "Text".to_string());
        let cs = CueSettings {
            region: Some("r1".to_string()),
            vertical: VerticalText::Lr,
            ..CueSettings::default()
        };
        doc.cues.push(sub);
        doc.cue_settings.push(cs);

        let output = write_document(&doc).expect("write_document");
        assert!(output.contains("REGION"));
        assert!(output.contains("id:r1"));
        assert!(output.contains("vertical:lr"));
        assert!(output.contains("region:r1"));
    }

    #[test]
    fn test_missing_webvtt_header() {
        let err = parse_document("Not a VTT file\n\n00:00:01.000 --> 00:00:04.000\nHi\n\n");
        assert!(err.is_err());
    }

    #[test]
    fn test_region_default_values() {
        let region = VttRegion::default();
        assert!((region.width_pct - 100.0).abs() < 0.01);
        assert_eq!(region.lines, 3);
        assert_eq!(region.scroll, RegionScroll::None);
    }

    #[test]
    fn test_parse_vtt_timestamp_short() {
        let ms = parse_vtt_timestamp_str("01:30.500").expect("short ts");
        assert_eq!(ms, 90_500);
    }

    #[test]
    fn test_parse_vtt_timestamp_long() {
        let ms = parse_vtt_timestamp_str("01:02:03.456").expect("long ts");
        assert_eq!(ms, 3_723_456);
    }

    #[test]
    fn test_cue_settings_default_horizontal() {
        let cs = CueSettings::default();
        assert_eq!(cs.vertical, VerticalText::Horizontal);
        assert!(cs.position_pct.is_none());
        assert!(cs.alignment.is_none());
    }

    #[test]
    fn test_parse_multiline_cue() {
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nLine one.\nLine two.\n\n";
        let subs = parse_webvtt(vtt).expect("multiline cue");
        assert_eq!(subs[0].text, "Line one.\nLine two.");
    }

    #[test]
    fn test_parse_empty_vtt() {
        let vtt = "WEBVTT\n\n";
        let subs = parse_webvtt(vtt).expect("empty vtt");
        assert!(subs.is_empty());
    }
}
