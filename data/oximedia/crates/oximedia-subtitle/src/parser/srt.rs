//! SubRip (SRT) subtitle parser.
//!
//! SRT is a simple subtitle format with the following structure:
//!
//! ```text
//! 1
//! 00:00:01,000 --> 00:00:04,000
//! This is the first subtitle.
//!
//! 2
//! 00:00:05,000 --> 00:00:08,000
//! This is the second subtitle.
//! It can span multiple lines.
//! ```

use crate::{Subtitle, SubtitleError, SubtitleResult};
use nom::{
    bytes::complete::{tag, take_until, take_while, take_while1},
    character::complete::{char, digit1, line_ending, not_line_ending},
    combinator::{map, map_res, opt},
    multi::many1,
    sequence::{preceded, separated_pair, terminated},
    IResult,
};

/// Parse SRT subtitle file.
///
/// # Errors
///
/// Returns error if the file is not valid SRT format.
pub fn parse(data: &[u8]) -> SubtitleResult<Vec<Subtitle>> {
    let text = String::from_utf8_lossy(data);
    parse_srt(&text)
}

/// Parse SRT subtitle from string.
///
/// # Errors
///
/// Returns error if parsing fails.
pub fn parse_srt(input: &str) -> SubtitleResult<Vec<Subtitle>> {
    // Normalize line endings
    let normalized = input.replace("\r\n", "\n");

    match parse_subtitle_file(&normalized) {
        Ok((_, subtitles)) => Ok(subtitles),
        Err(e) => Err(SubtitleError::ParseError(format!("SRT parse error: {e}"))),
    }
}

/// Check if text looks like SRT format.
#[must_use]
pub fn is_srt_format(text: &str) -> bool {
    // Look for typical SRT patterns: number, timestamp arrow, text
    let lines: Vec<&str> = text.lines().take(10).collect();

    for window in lines.windows(3) {
        if window[0].trim().chars().all(|c| c.is_ascii_digit())
            && window[1].contains("-->")
            && window[1].contains(':')
        {
            return true;
        }
    }

    false
}

/// Parse complete subtitle file.
fn parse_subtitle_file(input: &str) -> IResult<&str, Vec<Subtitle>> {
    // Skip BOM if present
    let mut input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut subtitles = Vec::new();

    loop {
        // Skip whitespace
        let (rest, _) = take_while(|c: char| c.is_whitespace())(input)?;
        input = rest;

        // Check if we've reached end of input
        if input.is_empty() {
            break;
        }

        // Try to parse an entry
        match parse_subtitle_entry(input) {
            Ok((rest, subtitle)) => {
                subtitles.push(subtitle);
                input = rest;
            }
            Err(_) => break,
        }
    }

    if subtitles.is_empty() {
        Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Many1,
        )))
    } else {
        Ok((input, subtitles))
    }
}

/// Parse a single subtitle entry.
fn parse_subtitle_entry(input: &str) -> IResult<&str, Subtitle> {
    let (input, _) = skip_empty_lines(input)?;

    let (input, _sequence) = digit1(input)?;
    let (input, _) = line_ending(input)?;
    let (input, (start, end)) = parse_timestamp_line(input)?;
    let (input, _) = line_ending(input)?;
    let (input, text) = subtitle_text(input)?;

    Ok((input, Subtitle::new(start, end, text)))
}

/// Skip empty lines and whitespace.
fn skip_empty_lines(input: &str) -> IResult<&str, ()> {
    let (input, _) = take_while(|c: char| c.is_whitespace())(input)?;
    Ok((input, ()))
}

/// Parse timestamp line (e.g., "00:00:01,000 --> 00:00:04,000").
fn parse_timestamp_line(input: &str) -> IResult<&str, (i64, i64)> {
    let (input, start) = timestamp(input)?;
    let (input, _) = tag(" --> ")(input)?;
    let (input, end) = timestamp(input)?;
    Ok((input, (start, end)))
}

/// Parse timestamp line with optional trailing content.
fn timestamp_line(input: &str) -> IResult<&str, (i64, i64)> {
    let (input, times) = parse_timestamp_line(input)?;
    // Try to parse optional trailing content
    let (input, _) = if let Ok((rest, _)) = tag::<_, _, nom::error::Error<_>>(" ")(input) {
        let (rest, _) = not_line_ending(rest)?;
        (rest, ())
    } else {
        (input, ())
    };
    Ok((input, times))
}

/// Parse a timestamp (e.g., "00:00:01,000").
fn timestamp(input: &str) -> IResult<&str, i64> {
    let (input, hours) = digit1(input)?;
    let (input, _) = char(':')(input)?;
    let (input, minutes) = digit1(input)?;
    let (input, _) = char(':')(input)?;
    let (input, seconds) = digit1(input)?;
    let (input, _) = char(',')(input)?;
    let (input, millis) = digit1(input)?;

    let result = parse_timestamp_parts(hours, minutes, seconds, millis)
        .map_err(|_| nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Fail)))?;

    Ok((input, result))
}

/// Parse timestamp components into milliseconds.
fn parse_timestamp_parts(
    hours: &str,
    minutes: &str,
    seconds: &str,
    millis: &str,
) -> Result<i64, std::num::ParseIntError> {
    let h: i64 = hours.parse()?;
    let m: i64 = minutes.parse()?;
    let s: i64 = seconds.parse()?;
    let ms: i64 = millis.parse()?;

    Ok(h * 3600000 + m * 60000 + s * 1000 + ms)
}

/// Parse subtitle text (until next entry or end of file).
fn subtitle_text(input: &str) -> IResult<&str, String> {
    let mut text = String::new();
    let mut remaining = input;

    #[allow(clippy::while_let_loop)]
    loop {
        // Try to parse a line
        match not_line_ending::<_, nom::error::Error<_>>(remaining) {
            Ok((rest, line)) => {
                if line.trim().is_empty() {
                    // Empty line marks end of subtitle
                    let (rest, _) =
                        line_ending::<_, nom::error::Error<_>>(rest).unwrap_or((rest, ""));
                    return Ok((rest, crate::text::decode_html_entities(&text)));
                }
                // Add line to text
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(line);
                // Try to consume line ending
                if let Ok((rest, _)) = line_ending::<_, nom::error::Error<_>>(rest) {
                    remaining = rest;
                } else {
                    remaining = rest;
                    break;
                }
            }
            Err(_) => break,
        }
    }

    if text.is_empty() {
        Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Many1,
        )))
    } else {
        Ok((remaining, crate::text::decode_html_entities(&text)))
    }
}

// ============================================================================
// HTML Tag Parsing and Formatting Preservation
// ============================================================================

/// Represents an inline formatting tag found in SRT text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SrtTag {
    /// Bold text: `<b>...</b>`
    Bold,
    /// Italic text: `<i>...</i>`
    Italic,
    /// Underline text: `<u>...</u>`
    Underline,
    /// Strikethrough text: `<s>...</s>`
    Strikethrough,
    /// Font with optional color/face/size: `<font color="#RRGGBB" face="..." size="...">`
    Font {
        /// Color in `#RRGGBB` format, if present.
        color: Option<String>,
        /// Font face name, if present.
        face: Option<String>,
        /// Font size, if present.
        size: Option<String>,
    },
}

/// A segment of SRT text that may carry inline formatting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtFormattedSegment {
    /// The plain text content.
    pub text: String,
    /// Whether this segment is bold.
    pub bold: bool,
    /// Whether this segment is italic.
    pub italic: bool,
    /// Whether this segment is underlined.
    pub underline: bool,
    /// Whether this segment is struck through.
    pub strikethrough: bool,
    /// Font color (hex `#RRGGBB`), if overridden.
    pub font_color: Option<String>,
    /// Font face, if overridden.
    pub font_face: Option<String>,
    /// Font size, if overridden.
    pub font_size: Option<String>,
}

impl SrtFormattedSegment {
    /// Create a plain (unstyled) segment.
    #[must_use]
    pub fn plain(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
            font_color: None,
            font_face: None,
            font_size: None,
        }
    }

    /// Returns `true` if this segment has any formatting.
    #[must_use]
    pub fn has_formatting(&self) -> bool {
        self.bold
            || self.italic
            || self.underline
            || self.strikethrough
            || self.font_color.is_some()
            || self.font_face.is_some()
            || self.font_size.is_some()
    }
}

/// Result of parsing SRT text with formatting preserved.
#[derive(Debug, Clone)]
pub struct SrtFormattedText {
    /// Segments with formatting metadata.
    pub segments: Vec<SrtFormattedSegment>,
}

impl SrtFormattedText {
    /// Extract the plain text content, stripping all formatting tags.
    #[must_use]
    pub fn plain_text(&self) -> String {
        let mut out = String::new();
        for seg in &self.segments {
            out.push_str(&seg.text);
        }
        out
    }

    /// Reconstruct the HTML-tagged SRT text from segments.
    #[must_use]
    pub fn to_tagged_string(&self) -> String {
        let mut out = String::new();
        for seg in &self.segments {
            let mut open_tags = Vec::new();
            if seg.bold {
                out.push_str("<b>");
                open_tags.push("</b>");
            }
            if seg.italic {
                out.push_str("<i>");
                open_tags.push("</i>");
            }
            if seg.underline {
                out.push_str("<u>");
                open_tags.push("</u>");
            }
            if seg.strikethrough {
                out.push_str("<s>");
                open_tags.push("</s>");
            }
            if seg.font_color.is_some() || seg.font_face.is_some() || seg.font_size.is_some() {
                out.push_str("<font");
                if let Some(ref c) = seg.font_color {
                    out.push_str(&format!(" color=\"{c}\""));
                }
                if let Some(ref f) = seg.font_face {
                    out.push_str(&format!(" face=\"{f}\""));
                }
                if let Some(ref s) = seg.font_size {
                    out.push_str(&format!(" size=\"{s}\""));
                }
                out.push('>');
                open_tags.push("</font>");
            }
            out.push_str(&seg.text);
            for close in open_tags.into_iter().rev() {
                out.push_str(close);
            }
        }
        out
    }
}

/// Strip HTML formatting tags from SRT text, returning plain text.
///
/// Handles `<b>`, `<i>`, `<u>`, `<s>`, `<font ...>` and their closing tags.
/// Nested tags are handled correctly.
#[must_use]
pub fn strip_html_tags(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c == '<' {
            // Consume the tag
            let mut tag_buf = String::new();
            chars.next(); // consume '<'
            while let Some(&tc) = chars.peek() {
                if tc == '>' {
                    chars.next(); // consume '>'
                    break;
                }
                tag_buf.push(tc);
                chars.next();
            }
            // We simply skip the tag content entirely
        } else {
            result.push(c);
            chars.next();
        }
    }

    result
}

/// Parse SRT text with HTML formatting tags, preserving formatting metadata.
///
/// This parses text containing `<b>`, `<i>`, `<u>`, `<s>`, and
/// `<font color="..." face="..." size="...">` tags into a vector of
/// formatted segments.
#[must_use]
pub fn parse_formatted_text(input: &str) -> SrtFormattedText {
    let mut segments = Vec::new();
    let mut tag_stack: Vec<SrtTag> = Vec::new();
    let mut current_text = String::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c == '<' {
            // Flush current text as a segment
            if !current_text.is_empty() {
                segments.push(build_segment(&current_text, &tag_stack));
                current_text.clear();
            }

            chars.next(); // consume '<'
            let mut tag_buf = String::new();
            while let Some(&tc) = chars.peek() {
                if tc == '>' {
                    chars.next();
                    break;
                }
                tag_buf.push(tc);
                chars.next();
            }

            let tag_lower = tag_buf.to_ascii_lowercase();
            let tag_trimmed = tag_lower.trim();

            if tag_trimmed == "b" {
                tag_stack.push(SrtTag::Bold);
            } else if tag_trimmed == "/b" {
                pop_tag(&mut tag_stack, |t| matches!(t, SrtTag::Bold));
            } else if tag_trimmed == "i" {
                tag_stack.push(SrtTag::Italic);
            } else if tag_trimmed == "/i" {
                pop_tag(&mut tag_stack, |t| matches!(t, SrtTag::Italic));
            } else if tag_trimmed == "u" {
                tag_stack.push(SrtTag::Underline);
            } else if tag_trimmed == "/u" {
                pop_tag(&mut tag_stack, |t| matches!(t, SrtTag::Underline));
            } else if tag_trimmed == "s" {
                tag_stack.push(SrtTag::Strikethrough);
            } else if tag_trimmed == "/s" {
                pop_tag(&mut tag_stack, |t| matches!(t, SrtTag::Strikethrough));
            } else if tag_trimmed.starts_with("font") {
                let font_tag = parse_font_tag(&tag_buf);
                tag_stack.push(font_tag);
            } else if tag_trimmed == "/font" {
                pop_tag(&mut tag_stack, |t| matches!(t, SrtTag::Font { .. }));
            }
            // Unknown tags are silently ignored
        } else {
            current_text.push(c);
            chars.next();
        }
    }

    // Flush remaining text
    if !current_text.is_empty() {
        segments.push(build_segment(&current_text, &tag_stack));
    }

    SrtFormattedText { segments }
}

/// Parse a `<font ...>` tag's attributes.
fn parse_font_tag(raw: &str) -> SrtTag {
    let mut color = None;
    let mut face = None;
    let mut size = None;

    // Simple attribute parser: look for key="value" or key='value' patterns
    let lower = raw.to_ascii_lowercase();
    if let Some(pos) = lower.find("color") {
        color = extract_attribute_value(raw, pos + 5);
    }
    if let Some(pos) = lower.find("face") {
        face = extract_attribute_value(raw, pos + 4);
    }
    if let Some(pos) = lower.find("size") {
        size = extract_attribute_value(raw, pos + 4);
    }

    SrtTag::Font { color, face, size }
}

/// Extract an attribute value after the `=` sign, supporting `"..."` and `'...'` delimiters.
fn extract_attribute_value(raw: &str, start_after: usize) -> Option<String> {
    let remainder = raw.get(start_after..)?;
    // Skip whitespace and '='
    let remainder = remainder.trim_start();
    let remainder = remainder.strip_prefix('=')?;
    let remainder = remainder.trim_start();

    if let Some(stripped) = remainder.strip_prefix('"') {
        let end = stripped.find('"')?;
        Some(stripped[..end].to_string())
    } else if let Some(stripped) = remainder.strip_prefix('\'') {
        let end = stripped.find('\'')?;
        Some(stripped[..end].to_string())
    } else {
        // Unquoted value: take until whitespace or end
        let end = remainder
            .find(|c: char| c.is_whitespace() || c == '>')
            .unwrap_or(remainder.len());
        if end == 0 {
            None
        } else {
            Some(remainder[..end].to_string())
        }
    }
}

/// Pop the most recent matching tag from the stack.
fn pop_tag<F: Fn(&SrtTag) -> bool>(stack: &mut Vec<SrtTag>, predicate: F) {
    if let Some(pos) = stack.iter().rposition(|t| predicate(t)) {
        stack.remove(pos);
    }
}

/// Build a formatted segment from current text and the active tag stack.
fn build_segment(text: &str, tags: &[SrtTag]) -> SrtFormattedSegment {
    let mut seg = SrtFormattedSegment::plain(text);
    for tag in tags {
        match tag {
            SrtTag::Bold => seg.bold = true,
            SrtTag::Italic => seg.italic = true,
            SrtTag::Underline => seg.underline = true,
            SrtTag::Strikethrough => seg.strikethrough = true,
            SrtTag::Font { color, face, size } => {
                if color.is_some() {
                    seg.font_color = color.clone();
                }
                if face.is_some() {
                    seg.font_face = face.clone();
                }
                if size.is_some() {
                    seg.font_size = size.clone();
                }
            }
        }
    }
    seg
}

/// Parse SRT and return subtitles with formatting preserved per-cue.
///
/// Each `Subtitle` in the result has its `text` field set to the plain text
/// (tags stripped), while the full formatting metadata can be retrieved
/// by calling [`parse_formatted_text`] on the original raw text.
///
/// # Errors
///
/// Returns error if the SRT data is not valid.
pub fn parse_with_formatting(data: &[u8]) -> SubtitleResult<Vec<(Subtitle, SrtFormattedText)>> {
    let text = String::from_utf8_lossy(data);
    parse_srt_with_formatting(&text)
}

/// Parse SRT string and return subtitles paired with their formatting data.
///
/// # Errors
///
/// Returns error if the SRT data is not valid.
pub fn parse_srt_with_formatting(input: &str) -> SubtitleResult<Vec<(Subtitle, SrtFormattedText)>> {
    let normalized = input.replace("\r\n", "\n");

    match parse_subtitle_file_raw(&normalized) {
        Ok((_, entries)) => {
            let mut results = Vec::with_capacity(entries.len());
            for (start, end, raw_text) in entries {
                let decoded = crate::text::decode_html_entities(&raw_text);
                let formatted = parse_formatted_text(&decoded);
                let plain = strip_html_tags(&decoded);
                let sub = Subtitle::new(start, end, plain);
                results.push((sub, formatted));
            }
            Ok(results)
        }
        Err(e) => Err(SubtitleError::ParseError(format!("SRT parse error: {e}"))),
    }
}

/// Parse subtitle file returning raw (unstripped) text per entry.
fn parse_subtitle_file_raw(input: &str) -> IResult<&str, Vec<(i64, i64, String)>> {
    let mut input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut entries = Vec::new();

    loop {
        let (rest, _) = take_while(|c: char| c.is_whitespace())(input)?;
        input = rest;
        if input.is_empty() {
            break;
        }
        match parse_subtitle_entry_raw(input) {
            Ok((rest, entry)) => {
                entries.push(entry);
                input = rest;
            }
            Err(_) => break,
        }
    }

    if entries.is_empty() {
        Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Many1,
        )))
    } else {
        Ok((input, entries))
    }
}

/// Parse a single subtitle entry, preserving raw text (with HTML tags).
fn parse_subtitle_entry_raw(input: &str) -> IResult<&str, (i64, i64, String)> {
    let (input, _) = skip_empty_lines(input)?;
    let (input, _sequence) = digit1(input)?;
    let (input, _) = line_ending(input)?;
    let (input, (start, end)) = parse_timestamp_line(input)?;
    let (input, _) = line_ending(input)?;
    let (input, text) = subtitle_text_raw(input)?;
    Ok((input, (start, end, text)))
}

/// Parse subtitle text without decoding HTML entities or stripping tags.
fn subtitle_text_raw(input: &str) -> IResult<&str, String> {
    let mut text = String::new();
    let mut remaining = input;

    #[allow(clippy::while_let_loop)]
    loop {
        match not_line_ending::<_, nom::error::Error<_>>(remaining) {
            Ok((rest, line)) => {
                if line.trim().is_empty() {
                    let (rest, _) =
                        line_ending::<_, nom::error::Error<_>>(rest).unwrap_or((rest, ""));
                    return Ok((rest, text));
                }
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(line);
                if let Ok((rest, _)) = line_ending::<_, nom::error::Error<_>>(rest) {
                    remaining = rest;
                } else {
                    remaining = rest;
                    break;
                }
            }
            Err(_) => break,
        }
    }

    if text.is_empty() {
        Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Many1,
        )))
    } else {
        Ok((remaining, text))
    }
}

/// Format milliseconds as SRT timestamp.
#[must_use]
pub fn format_timestamp(ms: i64) -> String {
    let hours = ms / 3600000;
    let minutes = (ms % 3600000) / 60000;
    let seconds = (ms % 60000) / 1000;
    let millis = ms % 1000;

    format!("{hours:02}:{minutes:02}:{seconds:02},{millis:03}")
}

/// Write subtitles in SRT format.
///
/// # Errors
///
/// Returns error if writing fails.
pub fn write(subtitles: &[Subtitle]) -> SubtitleResult<String> {
    let mut output = String::new();

    for (i, subtitle) in subtitles.iter().enumerate() {
        // Sequence number
        output.push_str(&format!("{}\n", i + 1));

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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_tags_basic() {
        assert_eq!(strip_html_tags("<b>Hello</b>"), "Hello");
        assert_eq!(strip_html_tags("<i>world</i>"), "world");
    }

    #[test]
    fn test_strip_html_tags_nested() {
        assert_eq!(strip_html_tags("<b><i>nested</i></b>"), "nested");
    }

    #[test]
    fn test_strip_html_tags_font() {
        let input = r##"<font color="#FF0000">Red text</font>"##;
        assert_eq!(strip_html_tags(input), "Red text");
    }

    #[test]
    fn test_strip_html_tags_no_tags() {
        assert_eq!(strip_html_tags("plain text"), "plain text");
    }

    #[test]
    fn test_strip_html_tags_mixed() {
        let input = "Hello <b>bold</b> and <i>italic</i> text";
        assert_eq!(strip_html_tags(input), "Hello bold and italic text");
    }

    #[test]
    fn test_parse_formatted_text_bold() {
        let ft = parse_formatted_text("<b>Hello</b>");
        assert_eq!(ft.segments.len(), 1);
        assert!(ft.segments[0].bold);
        assert_eq!(ft.segments[0].text, "Hello");
    }

    #[test]
    fn test_parse_formatted_text_italic() {
        let ft = parse_formatted_text("<i>Italic</i>");
        assert_eq!(ft.segments.len(), 1);
        assert!(ft.segments[0].italic);
        assert!(!ft.segments[0].bold);
    }

    #[test]
    fn test_parse_formatted_text_nested_bold_italic() {
        let ft = parse_formatted_text("<b><i>Both</i></b>");
        assert_eq!(ft.segments.len(), 1);
        assert!(ft.segments[0].bold);
        assert!(ft.segments[0].italic);
        assert_eq!(ft.segments[0].text, "Both");
    }

    #[test]
    fn test_parse_formatted_text_font_color() {
        let ft = parse_formatted_text(r##"<font color="#FF0000">Red</font>"##);
        assert_eq!(ft.segments.len(), 1);
        assert_eq!(ft.segments[0].font_color.as_deref(), Some("#FF0000"));
        assert_eq!(ft.segments[0].text, "Red");
    }

    #[test]
    fn test_parse_formatted_text_mixed_segments() {
        let ft = parse_formatted_text("Normal <b>bold</b> again");
        assert_eq!(ft.segments.len(), 3);
        assert!(!ft.segments[0].bold);
        assert_eq!(ft.segments[0].text, "Normal ");
        assert!(ft.segments[1].bold);
        assert_eq!(ft.segments[1].text, "bold");
        assert!(!ft.segments[2].bold);
        assert_eq!(ft.segments[2].text, " again");
    }

    #[test]
    fn test_parse_formatted_plain_text() {
        let ft = parse_formatted_text("Normal <b>bold</b> again");
        assert_eq!(ft.plain_text(), "Normal bold again");
    }

    #[test]
    fn test_to_tagged_string_roundtrip() {
        let ft = parse_formatted_text("<b>bold</b>");
        assert_eq!(ft.to_tagged_string(), "<b>bold</b>");
    }

    #[test]
    fn test_to_tagged_string_font_color() {
        let ft = parse_formatted_text(r##"<font color="#00FF00">green</font>"##);
        let tagged = ft.to_tagged_string();
        assert!(tagged.contains(r##"color="#00FF00""##));
        assert!(tagged.contains("green"));
    }

    #[test]
    fn test_formatted_segment_has_formatting() {
        let plain = SrtFormattedSegment::plain("hello");
        assert!(!plain.has_formatting());

        let mut bold = SrtFormattedSegment::plain("hello");
        bold.bold = true;
        assert!(bold.has_formatting());
    }

    #[test]
    fn test_parse_srt_with_formatting() {
        let srt = "1\n00:00:01,000 --> 00:00:04,000\n<b>Hello</b> <i>world</i>\n\n";
        let result = parse_srt_with_formatting(srt);
        assert!(result.is_ok());
        let entries = result.expect("should succeed in test");
        assert_eq!(entries.len(), 1);
        let (sub, fmt) = &entries[0];
        assert_eq!(sub.text, "Hello world");
        assert_eq!(fmt.segments.len(), 3);
        assert!(fmt.segments[0].bold);
        assert!(fmt.segments[2].italic);
    }

    #[test]
    fn test_parse_srt_with_formatting_font_tag() {
        let srt = "1\n00:00:01,000 --> 00:00:04,000\n<font color=\"#FF0000\">Red</font> normal\n\n";
        let result = parse_srt_with_formatting(srt);
        assert!(result.is_ok());
        let entries = result.expect("should succeed in test");
        let (sub, fmt) = &entries[0];
        assert_eq!(sub.text, "Red normal");
        assert_eq!(fmt.segments[0].font_color.as_deref(), Some("#FF0000"));
    }

    #[test]
    fn test_parse_formatted_text_font_face_size() {
        let ft = parse_formatted_text(r##"<font face="Arial" size="24">Styled</font>"##);
        assert_eq!(ft.segments.len(), 1);
        assert_eq!(ft.segments[0].font_face.as_deref(), Some("Arial"));
        assert_eq!(ft.segments[0].font_size.as_deref(), Some("24"));
    }

    #[test]
    fn test_underline_and_strikethrough() {
        let ft = parse_formatted_text("<u>under</u> <s>strike</s>");
        assert_eq!(ft.segments.len(), 3);
        assert!(ft.segments[0].underline);
        assert!(ft.segments[2].strikethrough);
    }

    #[test]
    fn test_strip_html_preserves_newlines() {
        let input = "<b>Line 1</b>\n<i>Line 2</i>";
        assert_eq!(strip_html_tags(input), "Line 1\nLine 2");
    }

    #[test]
    fn test_unclosed_tag_handling() {
        // Unclosed tags should not crash, just carry formatting through
        let ft = parse_formatted_text("<b>no close");
        assert_eq!(ft.segments.len(), 1);
        assert!(ft.segments[0].bold);
        assert_eq!(ft.segments[0].text, "no close");
    }

    #[test]
    fn test_empty_formatted_text() {
        let ft = parse_formatted_text("");
        assert!(ft.segments.is_empty());
        assert_eq!(ft.plain_text(), "");
    }
}
