//! Subtitle and timed-text resource management for IMF packages.
//!
//! IMF packages may include IMSC1 (TTML-based) subtitle tracks as separate
//! essence files. This module provides structures for managing subtitle resources
//! within an IMF composition, including timing, language mapping, and validation.
//!
//! # TTML / WebVTT parsing
//!
//! Two lightweight parsers are included:
//! - [`parse_ttml`] — parses a TTML/IMSC1 XML document and extracts [`TtmlCue`] entries
//! - [`parse_webvtt`] — parses a WebVTT text document and extracts [`TtmlCue`] entries
//!
//! Both parsers share the same output type so downstream code can work with either
//! subtitle format uniformly.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use quick_xml::events::Event;
use quick_xml::reader::Reader;
use quick_xml::XmlVersion;

use crate::ImfError;

/// Subtitle format type within an IMF package.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SubtitleFormat {
    /// IMSC1 (Internet Media Subtitles and Captions) - SMPTE ST 2067-2.
    Imsc1,
    /// IMSC1 Text profile.
    Imsc1Text,
    /// IMSC1 Image profile.
    Imsc1Image,
    /// SMPTE-TT (SMPTE Timed Text).
    SmpteTt,
}

impl fmt::Display for SubtitleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Imsc1 => write!(f, "IMSC1"),
            Self::Imsc1Text => write!(f, "IMSC1-Text"),
            Self::Imsc1Image => write!(f, "IMSC1-Image"),
            Self::SmpteTt => write!(f, "SMPTE-TT"),
        }
    }
}

/// Language code and metadata for a subtitle track.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubtitleLanguage {
    /// BCP-47 language tag (e.g., "en-US", "ja-JP").
    pub language_tag: String,
    /// Human-readable language name.
    pub display_name: String,
    /// Whether this is a forced narrative subtitle track.
    pub forced: bool,
    /// Whether this track is for hearing-impaired audiences (SDH/CC).
    pub hearing_impaired: bool,
}

impl SubtitleLanguage {
    /// Creates a new subtitle language entry.
    pub fn new(tag: &str, name: &str) -> Self {
        Self {
            language_tag: tag.to_string(),
            display_name: name.to_string(),
            forced: false,
            hearing_impaired: false,
        }
    }

    /// Sets the forced narrative flag.
    pub fn with_forced(mut self, forced: bool) -> Self {
        self.forced = forced;
        self
    }

    /// Sets the hearing-impaired flag.
    pub fn with_hearing_impaired(mut self, hi: bool) -> Self {
        self.hearing_impaired = hi;
        self
    }
}

impl fmt::Display for SubtitleLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.display_name, self.language_tag)?;
        if self.forced {
            write!(f, " [forced]")?;
        }
        if self.hearing_impaired {
            write!(f, " [HI]")?;
        }
        Ok(())
    }
}

/// Time range for a subtitle resource within the CPL timeline.
#[derive(Clone, Debug, PartialEq)]
pub struct SubtitleTimeRange {
    /// Entry point in edit units from the start of the resource.
    pub entry_point: u64,
    /// Duration in edit units.
    pub duration: u64,
    /// Edit rate numerator.
    pub edit_rate_num: u32,
    /// Edit rate denominator.
    pub edit_rate_den: u32,
}

impl SubtitleTimeRange {
    /// Creates a new time range.
    pub fn new(entry: u64, duration: u64, rate_num: u32, rate_den: u32) -> Self {
        Self {
            entry_point: entry,
            duration,
            edit_rate_num: rate_num,
            edit_rate_den: rate_den,
        }
    }

    /// Returns the duration in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        if self.edit_rate_num == 0 {
            return 0.0;
        }
        self.duration as f64 * self.edit_rate_den as f64 / self.edit_rate_num as f64
    }

    /// Returns the entry point in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn entry_point_seconds(&self) -> f64 {
        if self.edit_rate_num == 0 {
            return 0.0;
        }
        self.entry_point as f64 * self.edit_rate_den as f64 / self.edit_rate_num as f64
    }

    /// Returns the end point in edit units.
    pub fn end_point(&self) -> u64 {
        self.entry_point + self.duration
    }

    /// Checks if this range overlaps with another.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.entry_point < other.end_point() && other.entry_point < self.end_point()
    }
}

/// A subtitle resource entry within an IMF CPL.
#[derive(Clone, Debug)]
pub struct SubtitleResource {
    /// Unique resource identifier (UUID).
    pub id: String,
    /// Track file identifier referencing the MXF essence.
    pub track_file_id: String,
    /// Subtitle format.
    pub format: SubtitleFormat,
    /// Language metadata.
    pub language: SubtitleLanguage,
    /// Time range within the CPL.
    pub time_range: SubtitleTimeRange,
    /// Intrinsic duration of the source file in edit units.
    pub intrinsic_duration: u64,
    /// Hash of the track file (hex string).
    pub hash: Option<String>,
}

impl SubtitleResource {
    /// Creates a new subtitle resource.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: &str,
        track_file_id: &str,
        format: SubtitleFormat,
        language: SubtitleLanguage,
        time_range: SubtitleTimeRange,
        intrinsic_duration: u64,
    ) -> Self {
        Self {
            id: id.to_string(),
            track_file_id: track_file_id.to_string(),
            format,
            language,
            time_range,
            intrinsic_duration,
            hash: None,
        }
    }

    /// Sets the hash for this resource.
    pub fn with_hash(mut self, hash: &str) -> Self {
        self.hash = Some(hash.to_string());
        self
    }

    /// Validates that the time range is within the intrinsic duration.
    pub fn is_time_range_valid(&self) -> bool {
        self.time_range.end_point() <= self.intrinsic_duration
    }
}

/// Manages a collection of subtitle resources for an IMF composition.
#[derive(Clone, Debug)]
pub struct SubtitleResourceManager {
    /// All subtitle resources keyed by resource ID.
    resources: HashMap<String, SubtitleResource>,
}

impl SubtitleResourceManager {
    /// Creates a new empty manager.
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
        }
    }

    /// Adds a subtitle resource.
    pub fn add(&mut self, resource: SubtitleResource) {
        self.resources.insert(resource.id.clone(), resource);
    }

    /// Gets a resource by ID.
    pub fn get(&self, id: &str) -> Option<&SubtitleResource> {
        self.resources.get(id)
    }

    /// Removes a resource by ID.
    pub fn remove(&mut self, id: &str) -> Option<SubtitleResource> {
        self.resources.remove(id)
    }

    /// Returns the number of subtitle resources.
    pub fn len(&self) -> usize {
        self.resources.len()
    }

    /// Returns true if there are no subtitle resources.
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Returns all unique language tags.
    pub fn languages(&self) -> Vec<String> {
        let mut langs: Vec<String> = self
            .resources
            .values()
            .map(|r| r.language.language_tag.clone())
            .collect();
        langs.sort();
        langs.dedup();
        langs
    }

    /// Returns resources for a specific language tag.
    pub fn by_language(&self, lang_tag: &str) -> Vec<&SubtitleResource> {
        self.resources
            .values()
            .filter(|r| r.language.language_tag == lang_tag)
            .collect()
    }

    /// Returns resources of a specific format.
    pub fn by_format(&self, format: SubtitleFormat) -> Vec<&SubtitleResource> {
        self.resources
            .values()
            .filter(|r| r.format == format)
            .collect()
    }

    /// Validates all resources and returns a list of issues.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        for r in self.resources.values() {
            if !r.is_time_range_valid() {
                issues.push(format!(
                    "Resource {}: time range exceeds intrinsic duration ({} > {})",
                    r.id,
                    r.time_range.end_point(),
                    r.intrinsic_duration
                ));
            }
            if r.language.language_tag.is_empty() {
                issues.push(format!("Resource {}: missing language tag", r.id));
            }
        }
        issues
    }
}

impl Default for SubtitleResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---- TTML / WebVTT cue types and parsers ----

/// Style attributes extracted from a TTML `<style>` or inline `tts:*` attributes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TtmlStyle {
    /// Font size in pixels (or em-equivalent), if present.
    pub font_size: Option<f32>,
    /// RGB text colour encoded as `[R, G, B]` bytes, if present.
    pub color: Option<[u8; 3]>,
    /// Whether italic style is applied.
    pub italic: bool,
}

/// A single timed text cue, compatible with both TTML and WebVTT origins.
///
/// When parsed from TTML the `id` field is taken from the `xml:id` attribute of
/// the `<p>` element (or an empty string when absent). When parsed from WebVTT the
/// optional cue identifier line is used.
#[derive(Debug, Clone)]
pub struct TtmlCue {
    /// Presentation start time.
    pub begin: Duration,
    /// Presentation end time.
    pub end: Duration,
    /// Cue identifier (may be empty).
    pub id: String,
    /// Plain text content (HTML/TTML markup stripped).
    pub text: String,
    /// Optional style information extracted from inline attributes.
    pub style: Option<TtmlStyle>,
}

/// Parse a TTML/IMSC1 XML document (as raw bytes) into a sequence of [`TtmlCue`]s.
///
/// Only a minimal subset of the TTML specification is supported: `<p>` elements
/// anywhere inside the document tree with `begin`, `end`, and optional `xml:id`
/// attributes. The `tts:color`, `tts:fontSize`, and `tts:fontStyle` attributes are
/// parsed for basic style support.
///
/// # Errors
/// Returns [`ImfError::XmlError`] when the document is not well-formed XML, or
/// [`ImfError::InvalidStructure`] for unrecognisable timestamp syntax.
pub fn parse_ttml(xml_bytes: &[u8]) -> Result<Vec<TtmlCue>, ImfError> {
    let mut reader = Reader::from_reader(xml_bytes);
    reader.config_mut().trim_text(true);

    let mut cues: Vec<TtmlCue> = Vec::new();
    // Whether we are currently inside a <p> element collecting text.
    let mut in_paragraph = false;
    let mut current_begin: Option<Duration> = None;
    let mut current_end: Option<Duration> = None;
    let mut current_id = String::new();
    let mut current_text = String::new();
    let mut current_style: Option<TtmlStyle> = None;
    // Nesting depth inside a <p> element (for <span> children etc.)
    let mut para_depth: u32 = 0;

    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let qname = e.name();
                let local_name = local_name_of(qname.as_ref());
                if local_name == "p" && !in_paragraph {
                    // Parse begin/end/xml:id/style from <p> attributes
                    let (begin_ts, end_ts, id_attr, style, has_style) =
                        parse_p_attributes(e.attributes())?;

                    if let (Some(b), Some(ev)) = (begin_ts, end_ts) {
                        current_begin = Some(b);
                        current_end = Some(ev);
                        current_id = id_attr;
                        current_text.clear();
                        current_style = if has_style { Some(style) } else { None };
                        in_paragraph = true;
                        para_depth = 1;
                    }
                } else if in_paragraph {
                    para_depth += 1;
                }
            }
            Ok(Event::End(ref e)) => {
                let qname = e.name();
                let local_name = local_name_of(qname.as_ref());
                if in_paragraph {
                    para_depth -= 1;
                    if para_depth == 0 {
                        // Closing </p>
                        let begin = current_begin.take().unwrap_or(Duration::ZERO);
                        let end = current_end.take().unwrap_or(Duration::ZERO);
                        cues.push(TtmlCue {
                            begin,
                            end,
                            id: std::mem::take(&mut current_id),
                            text: current_text.trim().to_string(),
                            style: current_style.take(),
                        });
                        in_paragraph = false;
                    } else if local_name == "br" {
                        // Handle explicit </br> (unusual but valid)
                        current_text.push('\n');
                    }
                }
            }
            Ok(Event::Empty(ref e)) => {
                let qname = e.name();
                let local_name = local_name_of(qname.as_ref());
                if in_paragraph {
                    // Self-closing tags inside <p>, e.g. <br/>
                    if local_name == "br" {
                        current_text.push('\n');
                    }
                } else if local_name == "p" {
                    // Self-closing <p begin=... end=.../> (empty cue)
                    let (begin_ts, end_ts, id_attr, _, _) = parse_p_attributes(e.attributes())?;
                    if let (Some(b), Some(ev)) = (begin_ts, end_ts) {
                        cues.push(TtmlCue {
                            begin: b,
                            end: ev,
                            id: id_attr,
                            text: String::new(),
                            style: None,
                        });
                    }
                }
            }
            Ok(Event::Text(ref t)) => {
                if in_paragraph {
                    let text = t.decode().map_err(|e| ImfError::XmlError(e.to_string()))?;
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        if !current_text.is_empty() {
                            current_text.push(' ');
                        }
                        current_text.push_str(trimmed);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(ImfError::XmlError(e.to_string())),
            _ => {}
        }
        buf.clear();
    }

    Ok(cues)
}

/// Parse TTML `<p>` element attributes into timing, id, and style fields.
///
/// Returns `(begin, end, xml_id, style, has_style_attr)`.
fn parse_p_attributes(
    attributes: quick_xml::events::attributes::Attributes<'_>,
) -> Result<(Option<Duration>, Option<Duration>, String, TtmlStyle, bool), ImfError> {
    let mut begin_ts: Option<Duration> = None;
    let mut end_ts: Option<Duration> = None;
    let mut id_attr = String::new();
    let mut style = TtmlStyle::default();
    let mut has_style = false;

    for attr_result in attributes {
        let attr = attr_result.map_err(|e| ImfError::XmlError(e.to_string()))?;
        let key_bytes = attr.key.as_ref();
        let key = std::str::from_utf8(key_bytes).map_err(|e| ImfError::XmlError(e.to_string()))?;
        let value = attr
            .normalized_value(XmlVersion::Implicit1_0)
            .map_err(|e| ImfError::XmlError(e.to_string()))?;

        match key {
            "begin" => {
                begin_ts = Some(parse_ttml_timestamp(value.as_ref())?);
            }
            "end" => {
                end_ts = Some(parse_ttml_timestamp(value.as_ref())?);
            }
            "xml:id" | "id" => {
                id_attr = value.into_owned();
            }
            "tts:color" => {
                if let Some(rgb) = parse_ttml_color(value.as_ref()) {
                    style.color = Some(rgb);
                    has_style = true;
                }
            }
            "tts:fontSize" => {
                // Strip unit suffix before parsing: "18px" → 18.0, "80%" → 80.0
                let numeric = value
                    .trim_end_matches("px")
                    .trim_end_matches("em")
                    .trim_end_matches('%');
                if let Ok(fs) = numeric.parse::<f32>() {
                    style.font_size = Some(fs);
                    has_style = true;
                }
            }
            "tts:fontStyle" => {
                if value.as_ref() == "italic" {
                    style.italic = true;
                    has_style = true;
                }
            }
            _ => {}
        }
    }

    Ok((begin_ts, end_ts, id_attr, style, has_style))
}

/// Parse a WebVTT text document into a sequence of [`TtmlCue`]s.
///
/// Supports:
/// - The mandatory `WEBVTT` header line
/// - Optional cue ID lines (any line not containing `-->`)
/// - Cue timing lines in the format `HH:MM:SS.mmm --> HH:MM:SS.mmm`
/// - Multi-line cue text, terminated by a blank line
///
/// Style/positioning metadata after the `-->` timing is silently ignored.
///
/// # Errors
/// Returns [`ImfError::InvalidStructure`] for malformed timestamp syntax.
pub fn parse_webvtt(text: &str) -> Result<Vec<TtmlCue>, ImfError> {
    let mut cues: Vec<TtmlCue> = Vec::new();
    let mut lines = text.lines().peekable();

    // Verify WEBVTT header
    match lines.next() {
        Some(header) if header.starts_with("WEBVTT") => {}
        Some(other) => {
            return Err(ImfError::InvalidStructure(format!(
                "WebVTT document must start with 'WEBVTT', got: {other:?}"
            )));
        }
        None => return Ok(cues), // empty document
    }

    // Skip the rest of the header block (until first blank line)
    for line in lines.by_ref() {
        if line.trim().is_empty() {
            break;
        }
    }

    // Parse cue blocks.
    // Each iteration scans ahead for the next timing line, then collects cue text.
    loop {
        // Scan forward for a timing line, recording an optional cue ID.
        // Returns None when the input is exhausted.
        let timing = find_next_webvtt_timing(&mut lines)?;
        let (begin, end, candidate_id) = match timing {
            Some(t) => t,
            None => break,
        };

        // Collect cue text lines until a blank line or end of input.
        let mut text_lines: Vec<String> = Vec::new();
        loop {
            match lines.peek() {
                None => break,
                Some(l) if l.trim().is_empty() => {
                    let _ = lines.next(); // consume the blank line
                    break;
                }
                Some(_) => {
                    if let Some(l) = lines.next() {
                        text_lines.push(l.to_string());
                    }
                }
            }
        }

        cues.push(TtmlCue {
            begin,
            end,
            id: candidate_id.unwrap_or_default(),
            text: text_lines.join("\n"),
            style: None,
        });
    }

    Ok(cues)
}

/// Advance `lines` until a WebVTT timing line (`-->`) is found.
///
/// Returns `Some((begin, end, cue_id))` when a timing line is located, or
/// `None` when the iterator is exhausted without finding one.
/// A non-timing, non-blank line immediately before a timing line is treated
/// as the cue identifier.
fn find_next_webvtt_timing(
    lines: &mut std::iter::Peekable<std::str::Lines<'_>>,
) -> Result<Option<(Duration, Duration, Option<String>)>, ImfError> {
    let mut candidate_id: Option<String> = None;

    loop {
        match lines.next() {
            None => return Ok(None),
            Some(line) if line.trim().is_empty() => {
                // Blank line resets any pending cue ID
                candidate_id = None;
            }
            Some(line) if line.contains("-->") => {
                let (begin, end) = parse_webvtt_timing_line(line)?;
                return Ok(Some((begin, end, candidate_id)));
            }
            Some(line) => {
                // Non-blank, non-timing line: treat as a potential cue identifier
                candidate_id = Some(line.trim().to_string());
            }
        }
    }
}

/// Extract the local XML element name, stripping any namespace prefix.
///
/// For example `ttml:p` → `p`, `p` → `p`.
fn local_name_of(qualified: &[u8]) -> &str {
    let s = std::str::from_utf8(qualified).unwrap_or("");
    // quick-xml returns the local name (without prefix) from `name()`.
    // The colon check is a safety net for implementations that may include the prefix.
    match s.rfind(':') {
        Some(pos) => &s[pos + 1..],
        None => s,
    }
}

/// Parse a TTML clock-value timestamp string into a [`Duration`].
///
/// Supports the following TTML time expression formats (TTML 1.0 §7.4.1):
/// - `HH:MM:SS` — hours, minutes, seconds
/// - `HH:MM:SS.mmm` — with fractional seconds (millisecond precision)
/// - `S.mmm` — plain seconds with optional fraction (no colons)
///
/// SMPTE timecode format `HH:MM:SS:FF` is handled by treating the frames
/// component as sub-second (assuming 30 fps as a safe default).
///
/// # Errors
/// Returns [`ImfError::InvalidStructure`] for values that cannot be parsed.
fn parse_ttml_timestamp(s: &str) -> Result<Duration, ImfError> {
    let s = s.trim();

    // Count colons to determine format
    let colon_count = s.bytes().filter(|&b| b == b':').count();

    if colon_count >= 2 {
        // HH:MM:SS or HH:MM:SS.mmm or HH:MM:SS:FF
        let parts: Vec<&str> = s.splitn(4, ':').collect();
        let hours: u64 = parts[0].parse().map_err(|_| {
            ImfError::InvalidStructure(format!("Invalid TTML timestamp hours: {s}"))
        })?;
        let minutes: u64 = parts[1].parse().map_err(|_| {
            ImfError::InvalidStructure(format!("Invalid TTML timestamp minutes: {s}"))
        })?;
        // parts[2] may be "SS" or "SS.mmm"
        let (secs, millis) = parse_seconds_fraction(parts[2], s)?;
        let total_millis = hours * 3_600_000 + minutes * 60_000 + secs * 1000 + millis;

        // If there is a 4th part it is frames (SMPTE timecode) — approximate at 30 fps
        let frame_millis = if parts.len() == 4 {
            let frames: u64 = parts[3].parse().map_err(|_| {
                ImfError::InvalidStructure(format!("Invalid TTML timestamp frames: {s}"))
            })?;
            (frames * 1000) / 30
        } else {
            0
        };

        Ok(Duration::from_millis(total_millis + frame_millis))
    } else if colon_count == 1 {
        // MM:SS or MM:SS.mmm (WebVTT short form — should not appear in TTML but be lenient)
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        let minutes: u64 = parts[0].parse().map_err(|_| {
            ImfError::InvalidStructure(format!("Invalid TTML timestamp minutes: {s}"))
        })?;
        let (secs, millis) = parse_seconds_fraction(parts[1], s)?;
        let total_millis = minutes * 60_000 + secs * 1000 + millis;
        Ok(Duration::from_millis(total_millis))
    } else {
        // Plain seconds: S or S.mmm
        let (secs, millis) = parse_seconds_fraction(s, s)?;
        Ok(Duration::from_millis(secs * 1000 + millis))
    }
}

/// Split `"SS"` or `"SS.mmm"` into `(seconds, milliseconds)`.
fn parse_seconds_fraction(s: &str, context: &str) -> Result<(u64, u64), ImfError> {
    let err =
        || ImfError::InvalidStructure(format!("Invalid seconds value in timestamp: {context}"));
    if let Some((sec_str, frac_str)) = s.split_once('.') {
        let secs: u64 = sec_str.parse().map_err(|_| err())?;
        // Normalise fraction to milliseconds (pad/truncate to 3 digits)
        let millis = normalise_fraction_to_millis(frac_str)?;
        Ok((secs, millis))
    } else {
        let secs: u64 = s.trim().parse().map_err(|_| err())?;
        Ok((secs, 0))
    }
}

/// Normalise a decimal fraction string (e.g. `"5"`, `"50"`, `"500"`, `"5000"`) to
/// an integer number of **milliseconds**.
///
/// - `"5"`   → 500 ms  (pad right to 3 digits: "500")
/// - `"50"`  → 500 ms  ("500")
/// - `"500"` → 500 ms  (exact)
/// - `"5000"` → 500 ms  (truncate to 3 digits)
fn normalise_fraction_to_millis(frac: &str) -> Result<u64, ImfError> {
    // Truncate or pad to exactly 3 digits
    let padded: String = if frac.len() >= 3 {
        frac[..3].to_string()
    } else {
        format!("{:0<3}", frac)
    };
    padded
        .parse()
        .map_err(|_| ImfError::InvalidStructure(format!("Invalid fractional seconds: {frac}")))
}

/// Parse a WebVTT timing line of the form `HH:MM:SS.mmm --> HH:MM:SS.mmm [settings]`.
///
/// The optional settings portion after the second timestamp is silently discarded.
fn parse_webvtt_timing_line(line: &str) -> Result<(Duration, Duration), ImfError> {
    let err = || ImfError::InvalidStructure(format!("Invalid WebVTT timing line: {line}"));
    let (begin_part, rest) = line.split_once("-->").ok_or_else(err)?;
    // The rest may have trailing settings like `align:start position:10%`
    let end_part = rest.split_whitespace().next().ok_or_else(err)?;
    let begin = parse_ttml_timestamp(begin_part.trim())?;
    let end = parse_ttml_timestamp(end_part.trim())?;
    Ok((begin, end))
}

/// Parse a TTML `tts:color` attribute value into an RGB byte triple.
///
/// Supports:
/// - Named colours: `white`, `black`, `yellow`, `red`, `green`, `blue`, `cyan`, `magenta`
/// - Hex colours: `#RRGGBB` or `#RRGGBBAA` (alpha channel is ignored)
fn parse_ttml_color(s: &str) -> Option<[u8; 3]> {
    let s = s.trim().to_lowercase();
    match s.as_str() {
        "white" => Some([255, 255, 255]),
        "black" => Some([0, 0, 0]),
        "yellow" => Some([255, 255, 0]),
        "red" => Some([255, 0, 0]),
        "green" => Some([0, 128, 0]),
        "lime" => Some([0, 255, 0]),
        "blue" => Some([0, 0, 255]),
        "cyan" => Some([0, 255, 255]),
        "magenta" => Some([255, 0, 255]),
        _ if s.starts_with('#') && (s.len() == 7 || s.len() == 9) => {
            let hex = &s[1..];
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            Some([r, g, b])
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_resource(id: &str, lang: &str, entry: u64, dur: u64) -> SubtitleResource {
        SubtitleResource::new(
            id,
            &format!("track-{id}"),
            SubtitleFormat::Imsc1Text,
            SubtitleLanguage::new(lang, lang),
            SubtitleTimeRange::new(entry, dur, 24, 1),
            1000,
        )
    }

    #[test]
    fn test_subtitle_format_display() {
        assert_eq!(format!("{}", SubtitleFormat::Imsc1), "IMSC1");
        assert_eq!(format!("{}", SubtitleFormat::Imsc1Text), "IMSC1-Text");
        assert_eq!(format!("{}", SubtitleFormat::SmpteTt), "SMPTE-TT");
    }

    #[test]
    fn test_subtitle_language_basic() {
        let lang = SubtitleLanguage::new("en-US", "English");
        assert_eq!(lang.language_tag, "en-US");
        assert!(!lang.forced);
        assert!(!lang.hearing_impaired);
    }

    #[test]
    fn test_subtitle_language_flags() {
        let lang = SubtitleLanguage::new("en-US", "English")
            .with_forced(true)
            .with_hearing_impaired(true);
        assert!(lang.forced);
        assert!(lang.hearing_impaired);
        let display = format!("{lang}");
        assert!(display.contains("[forced]"));
        assert!(display.contains("[HI]"));
    }

    #[test]
    fn test_time_range_duration_seconds() {
        let tr = SubtitleTimeRange::new(0, 240, 24, 1);
        assert!((tr.duration_seconds() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_range_entry_point_seconds() {
        let tr = SubtitleTimeRange::new(48, 240, 24, 1);
        assert!((tr.entry_point_seconds() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_range_end_point() {
        let tr = SubtitleTimeRange::new(100, 200, 24, 1);
        assert_eq!(tr.end_point(), 300);
    }

    #[test]
    fn test_time_range_overlaps() {
        let a = SubtitleTimeRange::new(0, 100, 24, 1);
        let b = SubtitleTimeRange::new(50, 100, 24, 1);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_time_range_no_overlap() {
        let a = SubtitleTimeRange::new(0, 100, 24, 1);
        let b = SubtitleTimeRange::new(100, 50, 24, 1);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_resource_time_range_valid() {
        let r = make_resource("r1", "en", 0, 500);
        assert!(r.is_time_range_valid());
    }

    #[test]
    fn test_resource_time_range_invalid() {
        let r = make_resource("r1", "en", 900, 200); // 900+200=1100 > 1000
        assert!(!r.is_time_range_valid());
    }

    #[test]
    fn test_manager_add_get() {
        let mut mgr = SubtitleResourceManager::new();
        mgr.add(make_resource("r1", "en", 0, 100));
        assert_eq!(mgr.len(), 1);
        assert!(mgr.get("r1").is_some());
        assert!(mgr.get("r2").is_none());
    }

    #[test]
    fn test_manager_languages() {
        let mut mgr = SubtitleResourceManager::new();
        mgr.add(make_resource("r1", "en", 0, 100));
        mgr.add(make_resource("r2", "ja", 0, 100));
        mgr.add(make_resource("r3", "en", 100, 100));
        let langs = mgr.languages();
        assert_eq!(langs.len(), 2);
        assert!(langs.contains(&"en".to_string()));
        assert!(langs.contains(&"ja".to_string()));
    }

    #[test]
    fn test_manager_by_language() {
        let mut mgr = SubtitleResourceManager::new();
        mgr.add(make_resource("r1", "en", 0, 100));
        mgr.add(make_resource("r2", "ja", 0, 100));
        let en_resources = mgr.by_language("en");
        assert_eq!(en_resources.len(), 1);
    }

    #[test]
    fn test_manager_validate() {
        let mut mgr = SubtitleResourceManager::new();
        mgr.add(make_resource("ok", "en", 0, 500));
        mgr.add(make_resource("bad", "ja", 900, 200));
        let issues = mgr.validate();
        assert_eq!(issues.len(), 1);
        assert!(issues[0].contains("bad"));
    }

    #[test]
    fn test_manager_remove() {
        let mut mgr = SubtitleResourceManager::new();
        mgr.add(make_resource("r1", "en", 0, 100));
        assert!(mgr.remove("r1").is_some());
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_time_range_zero_rate() {
        let tr = SubtitleTimeRange::new(0, 100, 0, 1);
        assert_eq!(tr.duration_seconds(), 0.0);
        assert_eq!(tr.entry_point_seconds(), 0.0);
    }

    // ---- TTML / WebVTT parser tests ----

    /// Minimal valid 2-cue TTML document
    const TTML_BASIC: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<tt xmlns="http://www.w3.org/ns/ttml">
  <body>
    <div>
      <p begin="00:00:01.000" end="00:00:03.000" xml:id="c1">Hello world</p>
      <p begin="00:00:05.500" end="00:00:08.000" xml:id="c2">Second cue</p>
    </div>
  </body>
</tt>"#;

    #[test]
    fn test_parse_ttml_basic() {
        let cues = parse_ttml(TTML_BASIC.as_bytes()).expect("parse_ttml should succeed");
        assert_eq!(cues.len(), 2, "Expected 2 cues, got {}", cues.len());

        let c1 = &cues[0];
        assert_eq!(c1.id, "c1");
        assert_eq!(c1.text, "Hello world");
        assert_eq!(c1.begin, Duration::from_secs(1));
        assert_eq!(c1.end, Duration::from_secs(3));

        let c2 = &cues[1];
        assert_eq!(c2.id, "c2");
        assert_eq!(c2.text, "Second cue");
        assert_eq!(c2.begin, Duration::from_millis(5_500));
        assert_eq!(c2.end, Duration::from_secs(8));
    }

    #[test]
    fn test_parse_empty_ttml() {
        let empty_ttml = r#"<?xml version="1.0"?>
<tt xmlns="http://www.w3.org/ns/ttml"><body><div></div></body></tt>"#;
        let cues =
            parse_ttml(empty_ttml.as_bytes()).expect("parse_ttml on empty doc should succeed");
        assert!(cues.is_empty(), "Empty TTML document should yield no cues");
    }

    #[test]
    fn test_parse_ttml_style() {
        let ttml_with_style = r##"<?xml version="1.0"?>
<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling">
  <body>
    <div>
      <p begin="00:00:01.000" end="00:00:02.000" xml:id="s1"
         tts:color="yellow" tts:fontSize="18px" tts:fontStyle="italic">Styled cue</p>
    </div>
  </body>
</tt>"##;
        let cues = parse_ttml(ttml_with_style.as_bytes()).expect("should parse styled TTML");
        assert_eq!(cues.len(), 1);
        let style = cues[0].style.as_ref().expect("style should be present");
        assert_eq!(
            style.color,
            Some([255, 255, 0]),
            "yellow should decode to [255,255,0]"
        );
        assert!(style.italic, "italic style should be true");
        assert!(style.font_size.is_some(), "font_size should be parsed");
        let fs = style
            .font_size
            .expect("font_size should be Some after parsing tts:fontSize");
        assert!((fs - 18.0).abs() < 0.01, "font_size should be ~18.0");
    }

    #[test]
    fn test_parse_ttml_hex_color() {
        let ttml = r##"<?xml version="1.0"?>
<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling">
  <body>
    <div>
      <p begin="00:00:01.000" end="00:00:02.000" xml:id="h1"
         tts:color="#FF8000">Orange cue</p>
    </div>
  </body>
</tt>"##;
        let cues = parse_ttml(ttml.as_bytes()).expect("should parse hex-color TTML");
        let style = cues[0].style.as_ref().expect("style should be present");
        assert_eq!(style.color, Some([0xFF, 0x80, 0x00]));
    }

    #[test]
    fn test_parse_ttml_fractional_seconds() {
        // Plain fractional second timestamps (no colons)
        let ttml = r#"<?xml version="1.0"?>
<tt xmlns="http://www.w3.org/ns/ttml">
  <body><div>
    <p begin="3.5" end="7.25" xml:id="f1">Fractional</p>
  </div></body>
</tt>"#;
        let cues = parse_ttml(ttml.as_bytes()).expect("should parse fractional-second TTML");
        assert_eq!(cues[0].begin, Duration::from_millis(3_500));
        assert_eq!(cues[0].end, Duration::from_millis(7_250));
    }

    // ---- WebVTT tests ----

    const VTT_BASIC: &str = "WEBVTT\n\n\
c1\n\
00:00:01.000 --> 00:00:03.000\n\
Hello world\n\
\n\
c2\n\
00:00:05.500 --> 00:00:08.000\n\
Second cue\n";

    #[test]
    fn test_parse_webvtt_basic() {
        let cues = parse_webvtt(VTT_BASIC).expect("parse_webvtt should succeed");
        assert_eq!(cues.len(), 2, "Expected 2 cues, got {}", cues.len());

        assert_eq!(cues[0].id, "c1");
        assert_eq!(cues[0].text, "Hello world");
        assert_eq!(cues[0].begin, Duration::from_secs(1));
        assert_eq!(cues[0].end, Duration::from_secs(3));

        assert_eq!(cues[1].id, "c2");
        assert_eq!(cues[1].text, "Second cue");
        assert_eq!(cues[1].begin, Duration::from_millis(5_500));
        assert_eq!(cues[1].end, Duration::from_secs(8));
    }

    #[test]
    fn test_parse_webvtt_no_cue_ids() {
        let vtt = "WEBVTT\n\n\
00:00:01.000 --> 00:00:02.000\n\
Line one\n\
\n\
00:00:03.000 --> 00:00:04.000\n\
Line two\n";
        let cues = parse_webvtt(vtt).expect("should parse VTT without cue IDs");
        assert_eq!(cues.len(), 2);
        assert!(
            cues[0].id.is_empty(),
            "id should be empty when not provided"
        );
        assert_eq!(cues[0].text, "Line one");
    }

    #[test]
    fn test_parse_webvtt_multiline_cue_text() {
        let vtt = "WEBVTT\n\n\
00:00:01.000 --> 00:00:05.000\n\
First line\n\
Second line\n\
Third line\n";
        let cues = parse_webvtt(vtt).expect("should parse multi-line VTT cue");
        assert_eq!(cues.len(), 1);
        assert!(
            cues[0].text.contains("First line"),
            "text must contain first line"
        );
        assert!(
            cues[0].text.contains("Second line"),
            "text must contain second line"
        );
        assert!(
            cues[0].text.contains("Third line"),
            "text must contain third line"
        );
    }

    #[test]
    fn test_parse_webvtt_with_settings() {
        // Timing line with position/align settings — must be silently ignored
        let vtt = "WEBVTT\n\n\
00:00:01.000 --> 00:00:02.000 align:start position:10%\n\
Positioned cue\n";
        let cues = parse_webvtt(vtt).expect("should parse VTT with cue settings");
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].begin, Duration::from_secs(1));
        assert_eq!(cues[0].text, "Positioned cue");
    }

    #[test]
    fn test_parse_webvtt_empty() {
        let vtt = "WEBVTT\n\n";
        let cues = parse_webvtt(vtt).expect("empty WebVTT should succeed");
        assert!(cues.is_empty());
    }

    #[test]
    fn test_parse_webvtt_invalid_header() {
        let result = parse_webvtt("NOTWEBVTT\n\ncue text\n");
        assert!(
            result.is_err(),
            "Non-WEBVTT document should return an error"
        );
    }

    #[test]
    fn test_ttml_timestamp_hours() {
        // 1 hour, 30 minutes, 45 seconds, 500 ms
        let d = parse_ttml_timestamp("01:30:45.500").expect("valid timestamp");
        let expected_ms = 1 * 3_600_000 + 30 * 60_000 + 45 * 1000 + 500;
        assert_eq!(d, Duration::from_millis(expected_ms));
    }

    #[test]
    fn test_ttml_timestamp_plain_seconds() {
        let d = parse_ttml_timestamp("12.345").expect("plain seconds");
        assert_eq!(d, Duration::from_millis(12_345));
    }

    #[test]
    fn test_normalise_fraction_padding() {
        // "5" → 500ms (left-aligned, pad right)
        let d = parse_ttml_timestamp("00:00:01.5").expect("fraction pad");
        assert_eq!(d, Duration::from_millis(1_500));
    }

    #[test]
    fn test_ttml_color_named() {
        assert_eq!(parse_ttml_color("white"), Some([255, 255, 255]));
        assert_eq!(parse_ttml_color("BLACK"), Some([0, 0, 0]));
        assert_eq!(parse_ttml_color("Yellow"), Some([255, 255, 0]));
    }

    #[test]
    fn test_ttml_color_hex() {
        assert_eq!(parse_ttml_color("#ff0000"), Some([255, 0, 0]));
        // With alpha channel (#RRGGBBAA) — alpha ignored
        assert_eq!(parse_ttml_color("#00FF00FF"), Some([0, 255, 0]));
    }

    #[test]
    fn test_ttml_color_unknown() {
        assert_eq!(parse_ttml_color("chartreuse"), None);
        assert_eq!(parse_ttml_color("#GG0000"), None);
    }
}
