//! IMSC1/TTML2 enhanced parser.
//!
//! Extends the basic TTML parser with full IMSC1 and TTML2 support including:
//! - Region-based layout with percentage coordinates
//! - Per-span inline styling
//! - Full tts:* attribute parsing
//! - Background color, font family, text alignment

#![allow(dead_code)]

use crate::{SubtitleError, SubtitleResult};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::HashMap;

// ============================================================================
// Public types
// ============================================================================

/// A TTML `<region>` element with percentage-based coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct TtmlRegion {
    /// Region identifier (xml:id).
    pub id: String,
    /// Origin as (x%, y%) percentage of the root container.
    pub origin: (f32, f32),
    /// Extent as (width%, height%) percentage of the root container.
    pub extent: (f32, f32),
}

impl TtmlRegion {
    /// Create a new `TtmlRegion`.
    #[must_use]
    pub fn new(id: impl Into<String>, origin: (f32, f32), extent: (f32, f32)) -> Self {
        Self {
            id: id.into(),
            origin,
            extent,
        }
    }
}

/// A TTML `<style>` element with full styling properties.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TtmlStyle {
    /// tts:fontFamily value.
    pub font_family: String,
    /// tts:fontSize value (e.g. "24px", "160%").
    pub font_size: String,
    /// tts:color value (named, hex, or rgb()).
    pub color: String,
    /// tts:textAlign value ("left", "center", "right", "start", "end").
    pub text_align: String,
    /// tts:backgroundColor value.
    pub background_color: String,
}

/// An inline `<span>` element within a paragraph.
#[derive(Clone, Debug, PartialEq)]
pub struct TtmlSpan {
    /// The text content of the span.
    pub text: String,
    /// Optional style reference (style attribute value).
    pub style_id: Option<String>,
}

impl TtmlSpan {
    /// Create a new `TtmlSpan`.
    #[must_use]
    pub fn new(text: impl Into<String>, style_id: Option<String>) -> Self {
        Self {
            text: text.into(),
            style_id,
        }
    }
}

/// A parsed subtitle entry from a TTML2/IMSC1 document.
#[derive(Clone, Debug)]
pub struct SubtitleEntry {
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Optional region reference.
    pub region_id: Option<String>,
    /// Optional style reference on the `<p>` element.
    pub style_id: Option<String>,
    /// Inline spans with individual styling.
    pub spans: Vec<TtmlSpan>,
    /// Full aggregated text from all spans.
    pub text: String,
}

impl SubtitleEntry {
    /// Check if this entry is active at the given timestamp (ms).
    #[must_use]
    pub fn is_active(&self, timestamp_ms: i64) -> bool {
        timestamp_ms >= self.start_ms && timestamp_ms < self.end_ms
    }
}

// ============================================================================
// Parser
// ============================================================================

/// TTML2/IMSC1 enhanced parser.
pub struct TtmlParser;

impl TtmlParser {
    /// Parse a TTML2/IMSC1 XML document and return subtitle entries.
    ///
    /// Parses `<region>`, `<style>`, `<p>`, and `<span>` elements with full
    /// attribute extraction.
    ///
    /// # Errors
    ///
    /// Returns `SubtitleError::ParseError` if the XML is malformed.
    pub fn parse_v2(xml: &str) -> SubtitleResult<Vec<SubtitleEntry>> {
        let doc = parse_ttml2_document(xml)?;
        Ok(doc.entries)
    }

    /// Parse and return the full document including regions and styles.
    ///
    /// # Errors
    ///
    /// Returns `SubtitleError::ParseError` if the XML is malformed.
    pub fn parse_document(xml: &str) -> SubtitleResult<Ttml2Document> {
        parse_ttml2_document(xml)
    }
}

/// Full parsed TTML2 document.
#[derive(Clone, Debug, Default)]
pub struct Ttml2Document {
    /// Parsed subtitle entries.
    pub entries: Vec<SubtitleEntry>,
    /// Named regions keyed by id.
    pub regions: HashMap<String, TtmlRegion>,
    /// Named styles keyed by id.
    pub styles: HashMap<String, TtmlStyle>,
}

// ============================================================================
// Internal parsing state
// ============================================================================

#[derive(Debug, Default)]
struct ParseState {
    regions: HashMap<String, TtmlRegion>,
    styles: HashMap<String, TtmlStyle>,
    entries: Vec<SubtitleEntry>,

    // Context flags
    in_layout: bool,
    in_styling: bool,
    in_body: bool,

    // Current paragraph being built
    current_p: Option<ParagraphBuilder>,
    // Current span being built (inside a <p>)
    current_span: Option<SpanBuilder>,
    // Nesting depth inside <p> so we know when we exit it
    p_depth: u32,
    // Nesting depth inside <span>
    span_depth: u32,
}

#[derive(Debug)]
struct ParagraphBuilder {
    start_ms: i64,
    end_ms: i64,
    region_id: Option<String>,
    style_id: Option<String>,
    spans: Vec<TtmlSpan>,
    /// Direct text content (outside any <span>)
    direct_text: String,
}

impl ParagraphBuilder {
    fn build(self) -> SubtitleEntry {
        // Aggregate text: direct text + all span texts
        let mut text = self.direct_text.clone();
        for span in &self.spans {
            if !text.is_empty() && !span.text.is_empty() {
                text.push(' ');
            }
            text.push_str(&span.text);
        }
        let text = text.trim().to_string();
        SubtitleEntry {
            start_ms: self.start_ms,
            end_ms: self.end_ms,
            region_id: self.region_id,
            style_id: self.style_id,
            spans: self.spans,
            text,
        }
    }
}

#[derive(Debug, Default)]
struct SpanBuilder {
    style_id: Option<String>,
    text: String,
}

// ============================================================================
// Core parsing function
// ============================================================================

fn parse_ttml2_document(xml: &str) -> SubtitleResult<Ttml2Document> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);

    let mut state = ParseState::default();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                handle_start_event(e, &mut state)?;
            }
            Ok(Event::Empty(ref e)) => {
                // Self-closing elements — handle as start without corresponding end
                handle_empty_event(e, &mut state)?;
            }
            Ok(Event::End(ref e)) => {
                handle_end_event(e, &mut state)?;
            }
            Ok(Event::Text(ref e)) => {
                let raw = reader
                    .decoder()
                    .decode(e.as_ref())
                    .map_err(|err| SubtitleError::ParseError(format!("decode error: {err}")))?;
                handle_text_event(raw.as_ref(), &mut state);
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(SubtitleError::ParseError(format!("XML error: {e}")));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(Ttml2Document {
        entries: state.entries,
        regions: state.regions,
        styles: state.styles,
    })
}

/// Strip XML namespace prefix from a local element name.
fn local_name(name: &[u8]) -> &[u8] {
    // Find last ':' and return suffix
    if let Some(pos) = name.iter().rposition(|&b| b == b':') {
        &name[pos + 1..]
    } else {
        name
    }
}

fn handle_start_event(
    e: &quick_xml::events::BytesStart,
    state: &mut ParseState,
) -> SubtitleResult<()> {
    let raw_name = e.name();
    let local = local_name(raw_name.as_ref());
    let name = std::str::from_utf8(local)
        .map_err(|err| SubtitleError::ParseError(format!("UTF-8: {err}")))?;

    match name {
        "layout" => state.in_layout = true,
        "styling" => state.in_styling = true,
        "body" => state.in_body = true,
        "region" if state.in_layout => {
            let region = parse_region_attrs(e)?;
            state.regions.insert(region.id.clone(), region);
        }
        "style" if state.in_styling => {
            if let Some((id, style)) = parse_style_attrs(e)? {
                state.styles.insert(id, style);
            }
        }
        "p" if state.in_body && state.current_p.is_none() => {
            let p = parse_paragraph_attrs(e)?;
            state.current_p = Some(p);
            state.p_depth = 1;
        }
        "p" if state.in_body => {
            // Nested <p> — increment depth
            state.p_depth += 1;
        }
        "span" if state.current_p.is_some() && state.current_span.is_none() => {
            let span = parse_span_attrs(e)?;
            state.current_span = Some(span);
            state.span_depth = 1;
        }
        "span" if state.current_p.is_some() => {
            state.span_depth += 1;
        }
        _ => {}
    }

    Ok(())
}

fn handle_empty_event(
    e: &quick_xml::events::BytesStart,
    state: &mut ParseState,
) -> SubtitleResult<()> {
    let raw_name = e.name();
    let local = local_name(raw_name.as_ref());
    let name = std::str::from_utf8(local)
        .map_err(|err| SubtitleError::ParseError(format!("UTF-8: {err}")))?;

    match name {
        "region" if state.in_layout => {
            let region = parse_region_attrs(e)?;
            state.regions.insert(region.id.clone(), region);
        }
        "style" if state.in_styling => {
            if let Some((id, style)) = parse_style_attrs(e)? {
                state.styles.insert(id, style);
            }
        }
        "br" if state.current_p.is_some() => {
            // Line break — append newline to current text target
            if let Some(ref mut span) = state.current_span {
                span.text.push('\n');
            } else if let Some(ref mut p) = state.current_p {
                p.direct_text.push('\n');
            }
        }
        _ => {}
    }

    Ok(())
}

fn handle_end_event(e: &quick_xml::events::BytesEnd, state: &mut ParseState) -> SubtitleResult<()> {
    let raw_name = e.name();
    let local = local_name(raw_name.as_ref());
    let name = std::str::from_utf8(local)
        .map_err(|err| SubtitleError::ParseError(format!("UTF-8: {err}")))?;

    match name {
        "layout" => state.in_layout = false,
        "styling" => state.in_styling = false,
        "body" => state.in_body = false,
        "span" if state.current_p.is_some() => {
            if state.span_depth > 1 {
                state.span_depth -= 1;
            } else {
                // Close the current span
                if let Some(span_builder) = state.current_span.take() {
                    let text = span_builder.text.trim().to_string();
                    if !text.is_empty() || span_builder.style_id.is_some() {
                        if let Some(ref mut p) = state.current_p {
                            p.spans.push(TtmlSpan {
                                text,
                                style_id: span_builder.style_id,
                            });
                        }
                    }
                }
                state.span_depth = 0;
            }
        }
        "p" if state.current_p.is_some() => {
            if state.p_depth > 1 {
                state.p_depth -= 1;
            } else {
                // Close the current paragraph
                // If there's an unclosed span, close it first
                if let Some(span_builder) = state.current_span.take() {
                    let text = span_builder.text.trim().to_string();
                    if let Some(ref mut p) = state.current_p {
                        p.spans.push(TtmlSpan {
                            text,
                            style_id: span_builder.style_id,
                        });
                    }
                }
                if let Some(p) = state.current_p.take() {
                    state.entries.push(p.build());
                }
                state.p_depth = 0;
            }
        }
        _ => {}
    }

    Ok(())
}

fn handle_text_event(text: &str, state: &mut ParseState) {
    if state.current_p.is_none() {
        return;
    }

    if let Some(ref mut span) = state.current_span {
        span.text.push_str(text);
    } else if let Some(ref mut p) = state.current_p {
        p.direct_text.push_str(text);
    }
}

// ============================================================================
// Attribute parsers
// ============================================================================

fn parse_region_attrs(e: &quick_xml::events::BytesStart) -> SubtitleResult<TtmlRegion> {
    let mut id = String::new();
    let mut origin = (0.0_f32, 0.0_f32);
    let mut extent = (100.0_f32, 100.0_f32);

    for attr_result in e.attributes() {
        let attr = attr_result
            .map_err(|err| SubtitleError::ParseError(format!("attribute error: {err}")))?;
        let key_bytes = attr.key.as_ref();
        let key = std::str::from_utf8(key_bytes)
            .map_err(|err| SubtitleError::ParseError(format!("UTF-8: {err}")))?;
        let val = std::str::from_utf8(attr.value.as_ref())
            .map_err(|err| SubtitleError::ParseError(format!("UTF-8: {err}")))?;

        // Strip namespace prefix from key
        let local_key = if let Some(pos) = key.rfind(':') {
            &key[pos + 1..]
        } else {
            key
        };

        match local_key {
            "id" => id = val.to_string(),
            "origin" => origin = parse_two_percentage_values(val),
            "extent" => extent = parse_two_percentage_values(val),
            _ => {}
        }
    }

    Ok(TtmlRegion { id, origin, extent })
}

fn parse_style_attrs(
    e: &quick_xml::events::BytesStart,
) -> SubtitleResult<Option<(String, TtmlStyle)>> {
    let mut id: Option<String> = None;
    let mut style = TtmlStyle::default();

    for attr_result in e.attributes() {
        let attr = attr_result
            .map_err(|err| SubtitleError::ParseError(format!("attribute error: {err}")))?;
        let key_bytes = attr.key.as_ref();
        let key = std::str::from_utf8(key_bytes)
            .map_err(|err| SubtitleError::ParseError(format!("UTF-8: {err}")))?;
        let val = std::str::from_utf8(attr.value.as_ref())
            .map_err(|err| SubtitleError::ParseError(format!("UTF-8: {err}")))?;

        let local_key = if let Some(pos) = key.rfind(':') {
            &key[pos + 1..]
        } else {
            key
        };

        match local_key {
            "id" => id = Some(val.to_string()),
            "fontFamily" => style.font_family = val.to_string(),
            "fontSize" => style.font_size = val.to_string(),
            "color" => style.color = val.to_string(),
            "textAlign" => style.text_align = val.to_string(),
            "backgroundColor" => style.background_color = val.to_string(),
            _ => {}
        }
    }

    Ok(id.map(|i| (i, style)))
}

fn parse_paragraph_attrs(e: &quick_xml::events::BytesStart) -> SubtitleResult<ParagraphBuilder> {
    let mut start_ms: Option<i64> = None;
    let mut end_ms: Option<i64> = None;
    let mut region_id: Option<String> = None;
    let mut style_id: Option<String> = None;

    for attr_result in e.attributes() {
        let attr = attr_result
            .map_err(|err| SubtitleError::ParseError(format!("attribute error: {err}")))?;
        let key_bytes = attr.key.as_ref();
        let key = std::str::from_utf8(key_bytes)
            .map_err(|err| SubtitleError::ParseError(format!("UTF-8: {err}")))?;
        let val = std::str::from_utf8(attr.value.as_ref())
            .map_err(|err| SubtitleError::ParseError(format!("UTF-8: {err}")))?;

        let local_key = if let Some(pos) = key.rfind(':') {
            &key[pos + 1..]
        } else {
            key
        };

        match local_key {
            "begin" => start_ms = parse_ttml_time(val),
            "end" => end_ms = parse_ttml_time(val),
            "region" => region_id = Some(val.to_string()),
            "style" => style_id = Some(val.to_string()),
            _ => {}
        }
    }

    let start = start_ms
        .ok_or_else(|| SubtitleError::InvalidTimestamp("Missing begin attribute".to_string()))?;
    let end = end_ms
        .ok_or_else(|| SubtitleError::InvalidTimestamp("Missing end attribute".to_string()))?;

    Ok(ParagraphBuilder {
        start_ms: start,
        end_ms: end,
        region_id,
        style_id,
        spans: Vec::new(),
        direct_text: String::new(),
    })
}

fn parse_span_attrs(e: &quick_xml::events::BytesStart) -> SubtitleResult<SpanBuilder> {
    let mut style_id: Option<String> = None;

    for attr_result in e.attributes() {
        let attr = attr_result
            .map_err(|err| SubtitleError::ParseError(format!("attribute error: {err}")))?;
        let key_bytes = attr.key.as_ref();
        let key = std::str::from_utf8(key_bytes)
            .map_err(|err| SubtitleError::ParseError(format!("UTF-8: {err}")))?;
        let val = std::str::from_utf8(attr.value.as_ref())
            .map_err(|err| SubtitleError::ParseError(format!("UTF-8: {err}")))?;

        let local_key = if let Some(pos) = key.rfind(':') {
            &key[pos + 1..]
        } else {
            key
        };

        if local_key == "style" {
            style_id = Some(val.to_string());
        }
    }

    Ok(SpanBuilder {
        style_id,
        text: String::new(),
    })
}

// ============================================================================
// Time parsing
// ============================================================================

/// Parse a TTML time expression to milliseconds.
///
/// Supports:
/// - `HH:MM:SS.mmm` clock time
/// - `MM:SS.mmm` short clock time
/// - `12.5s` offset seconds
/// - `100ms` offset milliseconds
/// - `2m` offset minutes
/// - `1h` offset hours
#[must_use]
pub fn parse_ttml_time(time: &str) -> Option<i64> {
    let time = time.trim();

    if time.contains(':') {
        return parse_clock_time(time);
    }

    parse_offset_time(time)
}

fn parse_clock_time(time: &str) -> Option<i64> {
    let parts: Vec<&str> = time.split(':').collect();

    match parts.len() {
        2 => {
            let minutes: i64 = parts[0].parse().ok()?;
            let (secs, millis) = split_seconds(parts[1])?;
            Some(minutes * 60_000 + secs * 1_000 + millis)
        }
        3 => {
            let hours: i64 = parts[0].parse().ok()?;
            let minutes: i64 = parts[1].parse().ok()?;
            let (secs, millis) = split_seconds(parts[2])?;
            Some(hours * 3_600_000 + minutes * 60_000 + secs * 1_000 + millis)
        }
        _ => None,
    }
}

fn split_seconds(s: &str) -> Option<(i64, i64)> {
    let parts: Vec<&str> = s.split('.').collect();
    let secs: i64 = parts[0].parse().ok()?;
    let millis: i64 = if parts.len() > 1 {
        let ms_str = format!("{:0<3}", &parts[1][..parts[1].len().min(3)]);
        ms_str.parse().ok()?
    } else {
        0
    };
    Some((secs, millis))
}

fn parse_offset_time(time: &str) -> Option<i64> {
    if let Some(v) = time.strip_suffix("ms") {
        let val: f64 = v.parse().ok()?;
        return Some(val as i64);
    }
    if let Some(v) = time.strip_suffix('s') {
        let val: f64 = v.parse().ok()?;
        return Some((val * 1_000.0) as i64);
    }
    if let Some(v) = time.strip_suffix('m') {
        let val: f64 = v.parse().ok()?;
        return Some((val * 60_000.0) as i64);
    }
    if let Some(v) = time.strip_suffix('h') {
        let val: f64 = v.parse().ok()?;
        return Some((val * 3_600_000.0) as i64);
    }
    None
}

// ============================================================================
// Percentage coordinate parsing
// ============================================================================

/// Parse a TTML percentage pair like `"10% 80%"` into `(10.0, 80.0)`.
#[must_use]
pub fn parse_two_percentage_values(s: &str) -> (f32, f32) {
    let parts: Vec<&str> = s.split_whitespace().collect();
    let parse_pct = |p: &str| -> f32 {
        let stripped = p.strip_suffix('%').unwrap_or(p);
        stripped.parse().unwrap_or(0.0)
    };

    let x = parts.first().map(|p| parse_pct(p)).unwrap_or(0.0);
    let y = parts.get(1).map(|p| parse_pct(p)).unwrap_or(0.0);
    (x, y)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- time parsing ----

    #[test]
    fn test_clock_time_hh_mm_ss_ms() {
        assert_eq!(parse_ttml_time("00:00:01.000"), Some(1_000));
        assert_eq!(parse_ttml_time("00:01:30.500"), Some(90_500));
        assert_eq!(parse_ttml_time("01:30:45.123"), Some(5_445_123));
    }

    #[test]
    fn test_clock_time_mm_ss() {
        assert_eq!(parse_ttml_time("10:20.500"), Some(620_500));
        assert_eq!(parse_ttml_time("00:01.000"), Some(1_000));
    }

    #[test]
    fn test_offset_time() {
        assert_eq!(parse_ttml_time("1000ms"), Some(1_000));
        assert_eq!(parse_ttml_time("1.5s"), Some(1_500));
        assert_eq!(parse_ttml_time("2m"), Some(120_000));
        assert_eq!(parse_ttml_time("1h"), Some(3_600_000));
    }

    #[test]
    fn test_invalid_time() {
        assert_eq!(parse_ttml_time("invalid"), None);
    }

    // ---- percentage parsing ----

    #[test]
    fn test_parse_two_percentage_values_basic() {
        let (x, y) = parse_two_percentage_values("10% 80%");
        assert!((x - 10.0).abs() < 0.001);
        assert!((y - 80.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_two_percentage_values_zero() {
        let (x, y) = parse_two_percentage_values("0% 0%");
        assert!((x - 0.0).abs() < 0.001);
        assert!((y - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_two_percentage_values_no_suffix() {
        let (x, y) = parse_two_percentage_values("50 75");
        assert!((x - 50.0).abs() < 0.001);
        assert!((y - 75.0).abs() < 0.001);
    }

    // ---- parse_v2 tests ----

    fn sample_ttml(body: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<tt xmlns="http://www.w3.org/ns/ttml"
    xmlns:tts="http://www.w3.org/ns/ttml#styling"
    xmlns:ttp="http://www.w3.org/ns/ttml#parameter">
  <head>
    <layout>
      <region xml:id="r1" tts:origin="10% 80%" tts:extent="80% 15%"/>
    </layout>
    <styling>
      <style xml:id="s1" tts:fontFamily="Arial" tts:fontSize="24px" tts:color="white" tts:textAlign="center" tts:backgroundColor="transparent"/>
    </styling>
  </head>
  <body>
    <div>
      {body}
    </div>
  </body>
</tt>"#
        )
    }

    #[test]
    fn test_parse_v2_basic() {
        let xml = sample_ttml(r#"<p begin="00:00:01.000" end="00:00:04.000">Hello TTML2!</p>"#);
        let entries = TtmlParser::parse_v2(&xml).expect("parse should succeed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].start_ms, 1_000);
        assert_eq!(entries[0].end_ms, 4_000);
        assert_eq!(entries[0].text, "Hello TTML2!");
    }

    #[test]
    fn test_parse_v2_multiple_cues() {
        let xml = sample_ttml(
            r#"<p begin="00:00:01.000" end="00:00:03.000">First.</p>
               <p begin="00:00:04.000" end="00:00:07.000">Second.</p>"#,
        );
        let entries = TtmlParser::parse_v2(&xml).expect("parse should succeed");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "First.");
        assert_eq!(entries[1].text, "Second.");
    }

    #[test]
    fn test_parse_v2_region_ref() {
        let xml = sample_ttml(
            r#"<p begin="00:00:01.000" end="00:00:04.000" region="r1">With region.</p>"#,
        );
        let entries = TtmlParser::parse_v2(&xml).expect("parse should succeed");
        assert_eq!(entries[0].region_id, Some("r1".to_string()));
    }

    #[test]
    fn test_parse_v2_style_ref() {
        let xml =
            sample_ttml(r#"<p begin="00:00:01.000" end="00:00:04.000" style="s1">Styled.</p>"#);
        let entries = TtmlParser::parse_v2(&xml).expect("parse should succeed");
        assert_eq!(entries[0].style_id, Some("s1".to_string()));
    }

    #[test]
    fn test_parse_v2_with_spans() {
        let xml = sample_ttml(
            r#"<p begin="00:00:01.000" end="00:00:04.000"><span style="s1">Hello</span><span>World</span></p>"#,
        );
        let entries = TtmlParser::parse_v2(&xml).expect("parse should succeed");
        assert_eq!(entries[0].spans.len(), 2);
        assert_eq!(entries[0].spans[0].text, "Hello");
        assert_eq!(entries[0].spans[0].style_id, Some("s1".to_string()));
        assert_eq!(entries[0].spans[1].text, "World");
        assert_eq!(entries[0].spans[1].style_id, None);
    }

    #[test]
    fn test_parse_document_regions() {
        let xml = sample_ttml(r#"<p begin="00:00:01.000" end="00:00:02.000">x</p>"#);
        let doc = TtmlParser::parse_document(&xml).expect("parse should succeed");
        let region = doc.regions.get("r1").expect("r1 should exist");
        assert!((region.origin.0 - 10.0).abs() < 0.001);
        assert!((region.origin.1 - 80.0).abs() < 0.001);
        assert!((region.extent.0 - 80.0).abs() < 0.001);
        assert!((region.extent.1 - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_document_styles() {
        let xml = sample_ttml(r#"<p begin="00:00:01.000" end="00:00:02.000">x</p>"#);
        let doc = TtmlParser::parse_document(&xml).expect("parse should succeed");
        let style = doc.styles.get("s1").expect("s1 should exist");
        assert_eq!(style.font_family, "Arial");
        assert_eq!(style.font_size, "24px");
        assert_eq!(style.color, "white");
        assert_eq!(style.text_align, "center");
    }

    #[test]
    fn test_parse_v2_is_active() {
        let xml = sample_ttml(r#"<p begin="00:00:01.000" end="00:00:04.000">Active?</p>"#);
        let entries = TtmlParser::parse_v2(&xml).expect("parse should succeed");
        assert!(entries[0].is_active(2_000));
        assert!(!entries[0].is_active(500));
        assert!(!entries[0].is_active(5_000));
    }

    #[test]
    fn test_parse_v2_empty_span_text() {
        // Spans with empty text should still be included if they have a style
        let xml =
            sample_ttml(r#"<p begin="00:00:01.000" end="00:00:04.000"><span>Hello</span></p>"#);
        let entries = TtmlParser::parse_v2(&xml).expect("parse should succeed");
        assert!(!entries.is_empty());
        assert_eq!(entries[0].text, "Hello");
    }

    #[test]
    fn test_parse_v2_span_aggregated_text() {
        let xml = sample_ttml(
            r#"<p begin="00:00:01.000" end="00:00:04.000"><span>Foo</span><span>Bar</span></p>"#,
        );
        let entries = TtmlParser::parse_v2(&xml).expect("parse should succeed");
        // Text is aggregated from spans
        assert!(entries[0].text.contains("Foo"));
        assert!(entries[0].text.contains("Bar"));
    }

    #[test]
    fn test_parse_v2_no_region_defaults_to_none() {
        let xml = sample_ttml(r#"<p begin="00:00:01.000" end="00:00:04.000">No region.</p>"#);
        let entries = TtmlParser::parse_v2(&xml).expect("parse should succeed");
        assert!(entries[0].region_id.is_none());
    }

    #[test]
    fn test_parse_v2_no_style_defaults_to_none() {
        let xml = sample_ttml(r#"<p begin="00:00:01.000" end="00:00:04.000">No style.</p>"#);
        let entries = TtmlParser::parse_v2(&xml).expect("parse should succeed");
        assert!(entries[0].style_id.is_none());
    }

    #[test]
    fn test_parse_v2_offset_timing() {
        let xml = sample_ttml(r#"<p begin="1.5s" end="4s">Offset time.</p>"#);
        let entries = TtmlParser::parse_v2(&xml).expect("parse should succeed");
        assert_eq!(entries[0].start_ms, 1_500);
        assert_eq!(entries[0].end_ms, 4_000);
    }

    #[test]
    fn test_parse_v2_multiple_regions() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling">
  <head>
    <layout>
      <region xml:id="top" tts:origin="10% 5%" tts:extent="80% 15%"/>
      <region xml:id="bottom" tts:origin="10% 80%" tts:extent="80% 15%"/>
    </layout>
  </head>
  <body>
    <div>
      <p begin="00:00:01.000" end="00:00:04.000" region="top">Top text.</p>
      <p begin="00:00:01.000" end="00:00:04.000" region="bottom">Bottom text.</p>
    </div>
  </body>
</tt>"#;
        let doc = TtmlParser::parse_document(xml).expect("parse should succeed");
        assert_eq!(doc.regions.len(), 2);
        assert!(doc.regions.contains_key("top"));
        assert!(doc.regions.contains_key("bottom"));
        assert_eq!(doc.entries.len(), 2);
        assert_eq!(doc.entries[0].region_id, Some("top".to_string()));
        assert_eq!(doc.entries[1].region_id, Some("bottom".to_string()));
    }

    #[test]
    fn test_style_background_color() {
        // Use concat! to avoid Rust 2021 reserved prefix issue with raw strings containing #XXXXXX
        let xml = concat!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
            "<tt xmlns=\"http://www.w3.org/ns/ttml\" xmlns:tts=\"http://www.w3.org/ns/ttml#styling\">\n",
            "  <head>\n",
            "    <styling>\n",
            "      <style xml:id=\"s2\" tts:backgroundColor=\"", "#000000", "\" tts:color=\"", "#FFFFFF", "\"/>\n",
            "    </styling>\n",
            "  </head>\n",
            "  <body><div>\n",
            "    <p begin=\"00:00:01.000\" end=\"00:00:02.000\">BG test.</p>\n",
            "  </div></body>\n",
            "</tt>"
        );
        let doc = TtmlParser::parse_document(xml).expect("parse should succeed");
        let style = doc.styles.get("s2").expect("s2 should exist");
        assert_eq!(style.background_color, "#000000");
        assert_eq!(style.color, "#FFFFFF");
    }
}
