//! TTML (Timed Text Markup Language) subtitle parser and generator.
//!
//! TTML is a W3C standard XML-based format for timed text.
//!
//! ```xml
//! <?xml version="1.0" encoding="UTF-8"?>
//! <tt xmlns="http://www.w3.org/ns/ttml">
//!   <head>
//!     <styling>
//!       <style xml:id="s1" tts:color="white" tts:fontSize="24px"/>
//!     </styling>
//!   </head>
//!   <body>
//!     <div>
//!       <p begin="00:00:01.000" end="00:00:04.000">Hello world!</p>
//!     </div>
//!   </body>
//! </tt>
//! ```

use crate::style::{Alignment, Color, SubtitleStyle};
use crate::{Subtitle, SubtitleError, SubtitleResult};
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};
use std::collections::HashMap;
use std::io::Cursor;

/// TTML namespace.
const TTML_NS: &str = "http://www.w3.org/ns/ttml";
const TTS_NS: &str = "http://www.w3.org/ns/ttml#styling";
const TTP_NS: &str = "http://www.w3.org/ns/ttml#parameter";

/// TTML document with metadata and styles.
#[derive(Clone, Debug)]
pub struct TtmlDocument {
    /// Subtitle cues.
    pub subtitles: Vec<Subtitle>,
    /// Named styles.
    pub styles: HashMap<String, SubtitleStyle>,
    /// Document metadata.
    pub metadata: HashMap<String, String>,
}

/// Parse TTML subtitle file.
///
/// # Errors
///
/// Returns error if the file is not valid TTML format.
pub fn parse(data: &[u8]) -> SubtitleResult<Vec<Subtitle>> {
    let doc = parse_ttml(data)?;
    Ok(doc.subtitles)
}

/// Parse TTML document with full metadata.
///
/// # Errors
///
/// Returns error if parsing fails.
pub fn parse_ttml(data: &[u8]) -> SubtitleResult<TtmlDocument> {
    let mut reader = Reader::from_reader(data);
    reader.config_mut().trim_text(true);

    let mut styles = HashMap::new();
    let metadata = HashMap::new();
    let mut subtitles = Vec::new();
    let mut in_styling = false;
    let mut _in_metadata = false;
    let mut in_body = false;
    let mut current_p: Option<PElement> = None;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                let elem_name = e.name();
                let name = std::str::from_utf8(elem_name.as_ref())
                    .map_err(|e| SubtitleError::ParseError(format!("UTF-8 error: {e}")))?;

                match name {
                    "styling" => in_styling = true,
                    "metadata" => _in_metadata = true,
                    "body" => in_body = true,
                    "style" if in_styling => {
                        if let Some((id, style)) = parse_style_element(&e)? {
                            styles.insert(id, style);
                        }
                    }
                    "p" if in_body => {
                        current_p = Some(parse_p_element(&e)?);
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                let elem_name = e.name();
                let name = std::str::from_utf8(elem_name.as_ref())
                    .map_err(|e| SubtitleError::ParseError(format!("UTF-8 error: {e}")))?;

                match name {
                    "styling" => in_styling = false,
                    "metadata" => _in_metadata = false,
                    "body" => in_body = false,
                    "p" if in_body => {
                        if let Some(mut p) = current_p.take() {
                            // Apply style if referenced
                            if let Some(style_id) = &p.style_id {
                                if let Some(style) = styles.get(style_id) {
                                    p.subtitle.style = Some(style.clone());
                                }
                            }
                            subtitles.push(p.subtitle);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                if let Some(p) = &mut current_p {
                    let text = reader
                        .decoder()
                        .decode(e.as_ref())
                        .map_err(|e| SubtitleError::ParseError(format!("Decode error: {e}")))?;
                    if !p.subtitle.text.is_empty() {
                        p.subtitle.text.push(' ');
                    }
                    p.subtitle.text.push_str(&text);
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(SubtitleError::ParseError(format!("XML parse error: {e}"))),
            _ => {}
        }

        buf.clear();
    }

    Ok(TtmlDocument {
        subtitles,
        styles,
        metadata,
    })
}

/// Intermediate structure for `<p>` element parsing.
struct PElement {
    subtitle: Subtitle,
    style_id: Option<String>,
}

/// Parse `<style>` element.
fn parse_style_element(element: &BytesStart) -> SubtitleResult<Option<(String, SubtitleStyle)>> {
    let mut id = None;
    let mut style = SubtitleStyle::default();

    for attr in element.attributes() {
        let attr = attr.map_err(|e| SubtitleError::ParseError(format!("Attribute error: {e}")))?;
        let key = std::str::from_utf8(attr.key.as_ref())
            .map_err(|e| SubtitleError::ParseError(format!("UTF-8 error: {e}")))?;
        let value = std::str::from_utf8(&attr.value)
            .map_err(|e| SubtitleError::ParseError(format!("UTF-8 error: {e}")))?;

        match key {
            "xml:id" | "id" => id = Some(value.to_string()),
            "tts:color" | "color" => {
                if let Ok(color) = parse_ttml_color(value) {
                    style.primary_color = color;
                }
            }
            "tts:fontSize" | "fontSize" => {
                if let Some(size) = parse_font_size(value) {
                    style.font_size = size;
                }
            }
            "tts:fontFamily" | "fontFamily" => {
                // Font family is ignored - we use the provided font
            }
            "tts:fontWeight" | "fontWeight" => {
                style.font_weight = match value {
                    "bold" | "700" => crate::style::FontWeight::Bold,
                    "normal" | "400" => crate::style::FontWeight::Normal,
                    "extra-bold" | "800" => crate::style::FontWeight::ExtraBold,
                    "semi-bold" | "600" => crate::style::FontWeight::SemiBold,
                    "light" | "300" => crate::style::FontWeight::Light,
                    _ => crate::style::FontWeight::Normal,
                };
            }
            "tts:fontStyle" | "fontStyle" => {
                style.font_style = match value {
                    "italic" => crate::style::FontStyle::Italic,
                    "oblique" => crate::style::FontStyle::Oblique,
                    _ => crate::style::FontStyle::Normal,
                };
            }
            "tts:textAlign" | "textAlign" => {
                style.alignment = match value {
                    "left" | "start" => Alignment::Left,
                    "center" => Alignment::Center,
                    "right" | "end" => Alignment::Right,
                    _ => Alignment::Center,
                };
            }
            _ => {}
        }
    }

    if let Some(id) = id {
        Ok(Some((id, style)))
    } else {
        Ok(None)
    }
}

/// Parse `<p>` element.
fn parse_p_element(element: &BytesStart) -> SubtitleResult<PElement> {
    let mut begin: Option<i64> = None;
    let mut end: Option<i64> = None;
    let mut style_id: Option<String> = None;

    for attr in element.attributes() {
        let attr = attr.map_err(|e| SubtitleError::ParseError(format!("Attribute error: {e}")))?;
        let key = std::str::from_utf8(attr.key.as_ref())
            .map_err(|e| SubtitleError::ParseError(format!("UTF-8 error: {e}")))?;
        let value = std::str::from_utf8(&attr.value)
            .map_err(|e| SubtitleError::ParseError(format!("UTF-8 error: {e}")))?;

        match key {
            "begin" => begin = parse_ttml_time(value),
            "end" => end = parse_ttml_time(value),
            "style" => style_id = Some(value.to_string()),
            _ => {}
        }
    }

    let start_time = begin
        .ok_or_else(|| SubtitleError::InvalidTimestamp("Missing begin attribute".to_string()))?;
    let end_time =
        end.ok_or_else(|| SubtitleError::InvalidTimestamp("Missing end attribute".to_string()))?;

    Ok(PElement {
        subtitle: Subtitle::new(start_time, end_time, String::new()),
        style_id,
    })
}

/// Parse TTML time expression.
///
/// Supports:
/// - Clock time: `HH:MM:SS.mmm`
/// - Offset time: `12.5s`, `100ms`, `1h`, `30m`
fn parse_ttml_time(time: &str) -> Option<i64> {
    let time = time.trim();

    // Try clock time format (HH:MM:SS.mmm)
    if time.contains(':') {
        return parse_clock_time(time);
    }

    // Try offset time (e.g., "12.5s", "100ms")
    parse_offset_time(time)
}

/// Parse clock time format (HH:MM:SS.mmm or MM:SS.mmm).
fn parse_clock_time(time: &str) -> Option<i64> {
    let parts: Vec<&str> = time.split(':').collect();

    match parts.len() {
        2 => {
            // MM:SS or MM:SS.mmm
            let minutes: i64 = parts[0].parse().ok()?;
            let seconds_parts: Vec<&str> = parts[1].split('.').collect();
            let seconds: i64 = seconds_parts[0].parse().ok()?;
            let millis: i64 = if seconds_parts.len() > 1 {
                // Pad or truncate to 3 digits
                let ms_str = format!("{:0<3}", &seconds_parts[1][..seconds_parts[1].len().min(3)]);
                ms_str.parse().ok()?
            } else {
                0
            };
            Some(minutes * 60000 + seconds * 1000 + millis)
        }
        3 => {
            // HH:MM:SS or HH:MM:SS.mmm
            let hours: i64 = parts[0].parse().ok()?;
            let minutes: i64 = parts[1].parse().ok()?;
            let seconds_parts: Vec<&str> = parts[2].split('.').collect();
            let seconds: i64 = seconds_parts[0].parse().ok()?;
            let millis: i64 = if seconds_parts.len() > 1 {
                let ms_str = format!("{:0<3}", &seconds_parts[1][..seconds_parts[1].len().min(3)]);
                ms_str.parse().ok()?
            } else {
                0
            };
            Some(hours * 3600000 + minutes * 60000 + seconds * 1000 + millis)
        }
        _ => None,
    }
}

/// Parse offset time (e.g., "12.5s", "100ms", "1h", "30m").
fn parse_offset_time(time: &str) -> Option<i64> {
    let time = time.trim();

    if let Some(value_str) = time.strip_suffix("ms") {
        // Milliseconds
        let value: f64 = value_str.parse().ok()?;
        return Some(value as i64);
    }

    if let Some(value_str) = time.strip_suffix('s') {
        // Seconds
        let value: f64 = value_str.parse().ok()?;
        return Some((value * 1000.0) as i64);
    }

    if let Some(value_str) = time.strip_suffix('m') {
        // Minutes
        let value: f64 = value_str.parse().ok()?;
        return Some((value * 60000.0) as i64);
    }

    if let Some(value_str) = time.strip_suffix('h') {
        // Hours
        let value: f64 = value_str.parse().ok()?;
        return Some((value * 3600000.0) as i64);
    }

    None
}

/// Parse TTML color.
fn parse_ttml_color(color: &str) -> Result<Color, SubtitleError> {
    let color = color.trim();

    // Named colors
    match color.to_lowercase().as_str() {
        "white" => return Ok(Color::white()),
        "black" => return Ok(Color::black()),
        "red" => return Ok(Color::rgb(255, 0, 0)),
        "green" => return Ok(Color::rgb(0, 255, 0)),
        "blue" => return Ok(Color::rgb(0, 0, 255)),
        "yellow" => return Ok(Color::rgb(255, 255, 0)),
        "cyan" => return Ok(Color::rgb(0, 255, 255)),
        "magenta" => return Ok(Color::rgb(255, 0, 255)),
        _ => {}
    }

    // Hex color
    if color.starts_with('#') {
        return Color::from_hex(color);
    }

    // RGB/RGBA function
    if color.starts_with("rgb") {
        return parse_rgb_color(color);
    }

    Err(SubtitleError::InvalidColor(color.to_string()))
}

/// Parse RGB/RGBA color function.
fn parse_rgb_color(color: &str) -> Result<Color, SubtitleError> {
    let color = color.trim();

    let is_rgba = color.starts_with("rgba");
    let start = if is_rgba { 5 } else { 4 };

    let inner = color
        .get(start..color.len().saturating_sub(1))
        .ok_or_else(|| SubtitleError::InvalidColor(color.to_string()))?;

    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();

    if (is_rgba && parts.len() != 4) || (!is_rgba && parts.len() != 3) {
        return Err(SubtitleError::InvalidColor(color.to_string()));
    }

    let r = parts[0]
        .parse::<u8>()
        .map_err(|_| SubtitleError::InvalidColor(color.to_string()))?;
    let g = parts[1]
        .parse::<u8>()
        .map_err(|_| SubtitleError::InvalidColor(color.to_string()))?;
    let b = parts[2]
        .parse::<u8>()
        .map_err(|_| SubtitleError::InvalidColor(color.to_string()))?;

    let a = if is_rgba {
        let alpha_f = parts[3]
            .parse::<f32>()
            .map_err(|_| SubtitleError::InvalidColor(color.to_string()))?;
        (alpha_f * 255.0) as u8
    } else {
        255
    };

    Ok(Color::new(r, g, b, a))
}

/// Parse font size (e.g., "24px", "1.5em", "120%").
fn parse_font_size(size: &str) -> Option<f32> {
    let size = size.trim();

    if let Some(px) = size.strip_suffix("px") {
        return px.parse().ok();
    }

    if let Some(em) = size.strip_suffix("em") {
        let value: f32 = em.parse().ok()?;
        // Assume base font size of 24px
        return Some(value * 24.0);
    }

    if let Some(pct) = size.strip_suffix('%') {
        let value: f32 = pct.parse().ok()?;
        // Assume base font size of 24px
        return Some(value * 24.0 / 100.0);
    }

    // Try plain number (assume pixels)
    size.parse().ok()
}

/// Format milliseconds as TTML clock time.
#[must_use]
pub fn format_timestamp(ms: i64) -> String {
    let hours = ms / 3600000;
    let minutes = (ms % 3600000) / 60000;
    let seconds = (ms % 60000) / 1000;
    let millis = ms % 1000;

    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

/// Write subtitles in TTML format.
///
/// # Errors
///
/// Returns error if writing fails.
pub fn write(subtitles: &[Subtitle]) -> SubtitleResult<String> {
    let mut writer = Writer::new(Cursor::new(Vec::new()));

    // XML declaration
    writer
        .write_event(Event::Decl(quick_xml::events::BytesDecl::new(
            "1.0",
            Some("UTF-8"),
            None,
        )))
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;

    // Root <tt> element
    let mut tt_elem = BytesStart::new("tt");
    tt_elem.push_attribute(("xmlns", TTML_NS));
    tt_elem.push_attribute(("xmlns:tts", TTS_NS));
    tt_elem.push_attribute(("xmlns:ttp", TTP_NS));
    tt_elem.push_attribute(("xml:lang", "en"));
    writer
        .write_event(Event::Start(tt_elem))
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;

    // <head> with default style
    writer
        .write_event(Event::Start(BytesStart::new("head")))
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;

    writer
        .write_event(Event::Start(BytesStart::new("styling")))
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;

    let mut style_elem = BytesStart::new("style");
    style_elem.push_attribute(("xml:id", "default"));
    style_elem.push_attribute(("tts:color", "white"));
    style_elem.push_attribute(("tts:fontSize", "24px"));
    style_elem.push_attribute(("tts:fontFamily", "sans-serif"));
    style_elem.push_attribute(("tts:textAlign", "center"));
    writer
        .write_event(Event::Empty(style_elem))
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;

    writer
        .write_event(Event::End(BytesEnd::new("styling")))
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;

    writer
        .write_event(Event::End(BytesEnd::new("head")))
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;

    // <body>
    writer
        .write_event(Event::Start(BytesStart::new("body")))
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;

    writer
        .write_event(Event::Start(BytesStart::new("div")))
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;

    // Write subtitle cues
    for subtitle in subtitles {
        let mut p_elem = BytesStart::new("p");
        p_elem.push_attribute(("begin", format_timestamp(subtitle.start_time).as_str()));
        p_elem.push_attribute(("end", format_timestamp(subtitle.end_time).as_str()));
        p_elem.push_attribute(("style", "default"));

        writer
            .write_event(Event::Start(p_elem))
            .map_err(|e| SubtitleError::IoError(e.to_string()))?;

        writer
            .write_event(Event::Text(BytesText::new(&subtitle.text)))
            .map_err(|e| SubtitleError::IoError(e.to_string()))?;

        writer
            .write_event(Event::End(BytesEnd::new("p")))
            .map_err(|e| SubtitleError::IoError(e.to_string()))?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("div")))
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;

    writer
        .write_event(Event::End(BytesEnd::new("body")))
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;

    writer
        .write_event(Event::End(BytesEnd::new("tt")))
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;

    let result = writer.into_inner().into_inner();
    String::from_utf8(result).map_err(|e| SubtitleError::IoError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_clock_time() {
        assert_eq!(parse_clock_time("00:00:01.000"), Some(1000));
        assert_eq!(parse_clock_time("00:01:30.500"), Some(90500));
        assert_eq!(parse_clock_time("01:30:45.123"), Some(5445123));
        assert_eq!(parse_clock_time("10:20.500"), Some(620500));
    }

    #[test]
    fn test_parse_offset_time() {
        assert_eq!(parse_offset_time("1000ms"), Some(1000));
        assert_eq!(parse_offset_time("1.5s"), Some(1500));
        assert_eq!(parse_offset_time("2m"), Some(120000));
        assert_eq!(parse_offset_time("1h"), Some(3600000));
    }

    #[test]
    fn test_parse_ttml_color() {
        assert_eq!(
            parse_ttml_color("white").expect("should succeed in test"),
            Color::white()
        );
        assert_eq!(
            parse_ttml_color("#FFFFFF").expect("should succeed in test"),
            Color::white()
        );
        assert_eq!(
            parse_ttml_color("rgb(255,255,255)").expect("should succeed in test"),
            Color::white()
        );
        assert_eq!(
            parse_ttml_color("rgba(255,255,255,1.0)").expect("should succeed in test"),
            Color::white()
        );
    }

    #[test]
    fn test_parse_font_size() {
        assert_eq!(parse_font_size("24px"), Some(24.0));
        assert_eq!(parse_font_size("1.5em"), Some(36.0));
        assert_eq!(parse_font_size("150%"), Some(36.0));
    }

    #[test]
    fn test_format_timestamp() {
        assert_eq!(format_timestamp(1000), "00:00:01.000");
        assert_eq!(format_timestamp(90500), "00:01:30.500");
        assert_eq!(format_timestamp(5445123), "01:30:45.123");
    }

    #[test]
    fn test_parse_simple_ttml() {
        let ttml = r#"<?xml version="1.0" encoding="UTF-8"?>
<tt xmlns="http://www.w3.org/ns/ttml">
  <body>
    <div>
      <p begin="00:00:01.000" end="00:00:04.000">Hello world!</p>
    </div>
  </body>
</tt>"#;

        let doc = parse_ttml(ttml.as_bytes()).expect("should succeed in test");
        assert_eq!(doc.subtitles.len(), 1);
        assert_eq!(doc.subtitles[0].start_time, 1000);
        assert_eq!(doc.subtitles[0].end_time, 4000);
        assert_eq!(doc.subtitles[0].text, "Hello world!");
    }

    #[test]
    fn test_write_ttml() {
        let subtitles = vec![
            Subtitle::new(1000, 4000, "Hello world!".to_string()),
            Subtitle::new(5000, 8000, "Second subtitle".to_string()),
        ];

        let output = write(&subtitles).expect("should succeed in test");
        assert!(output.contains("<tt"));
        assert!(output.contains("begin=\"00:00:01.000\""));
        assert!(output.contains("end=\"00:00:04.000\""));
        assert!(output.contains("Hello world!"));
        assert!(output.contains("Second subtitle"));
    }
}
