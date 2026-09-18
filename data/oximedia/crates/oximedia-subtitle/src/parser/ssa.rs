//! SubStation Alpha (SSA) and Advanced SubStation Alpha (ASS) parser.
//!
//! SSA/ASS is a complex subtitle format with extensive styling support.
//!
//! ```text
//! [Script Info]
//! Title: Movie Subtitles
//! ScriptType: v4.00+
//!
//! [V4+ Styles]
//! Format: Name, Fontname, Fontsize, PrimaryColour, ...
//! Style: Default,Arial,48,&H00FFFFFF,&H000000FF,...
//!
//! [Events]
//! Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
//! Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Hello world!
//! ```

use crate::ssa_style_cache::AssStyleCache;
use crate::style::{Alignment, Color, FontWeight, OutlineStyle, Position, ShadowStyle};
use crate::{Subtitle, SubtitleError, SubtitleResult, SubtitleStyle};
use std::collections::HashMap;

/// SSA/ASS subtitle file.
#[derive(Clone, Debug)]
pub struct AssFile {
    /// Script metadata.
    pub script_info: HashMap<String, String>,
    /// Named styles.
    pub styles: HashMap<String, SubtitleStyle>,
    /// Subtitle events.
    pub events: Vec<Subtitle>,
    /// Pre-built style cache for O(1) dialogue-line lookup.
    ///
    /// Populated automatically from `styles` during parsing.
    pub style_cache: AssStyleCache,
}

/// Parse SSA/ASS subtitle file.
///
/// # Errors
///
/// Returns error if the file is not valid SSA/ASS format.
pub fn parse(data: &[u8]) -> SubtitleResult<Vec<Subtitle>> {
    let text = String::from_utf8_lossy(data);
    let file = parse_ass(&text)?;
    Ok(file.events)
}

/// Parse SSA/ASS subtitle from string.
///
/// # Errors
///
/// Returns error if parsing fails.
pub fn parse_ass(input: &str) -> SubtitleResult<AssFile> {
    let normalized = input.replace("\r\n", "\n");

    let mut script_info = HashMap::new();
    let mut styles = HashMap::new();
    let mut events = Vec::new();

    let mut current_section = String::new();
    let mut style_format = Vec::new();
    let mut event_format = Vec::new();

    for line in normalized.lines() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with(';') {
            continue;
        }

        // Section headers
        if line.starts_with('[') && line.ends_with(']') {
            current_section = line[1..line.len() - 1].to_string();
            continue;
        }

        match current_section.as_str() {
            "Script Info" => {
                if let Some((key, value)) = parse_key_value(line) {
                    script_info.insert(key, value);
                }
            }
            "V4+ Styles" | "V4 Styles" => {
                if line.starts_with("Format:") {
                    style_format = parse_format_line(line);
                } else if line.starts_with("Style:") {
                    if let Some(style) = parse_style_line(line, &style_format) {
                        styles.insert(style.0, style.1);
                    }
                }
            }
            "Events" => {
                if line.starts_with("Format:") {
                    event_format = parse_format_line(line);
                } else if line.starts_with("Dialogue:") || line.starts_with("Comment:") {
                    if let Some(event) = parse_event_line(line, &event_format, &styles) {
                        events.push(event);
                    }
                }
            }
            _ => {}
        }
    }

    // Build the style cache from the fully-parsed styles map so that callers
    // can do O(1) style lookups without re-scanning the HashMap.
    let style_cache = AssStyleCache::from_map(styles.clone());

    Ok(AssFile {
        script_info,
        styles,
        events,
        style_cache,
    })
}

/// Parse key-value pair.
fn parse_key_value(line: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = line.splitn(2, ':').collect();
    if parts.len() == 2 {
        Some((parts[0].trim().to_string(), parts[1].trim().to_string()))
    } else {
        None
    }
}

/// Parse format line.
fn parse_format_line(line: &str) -> Vec<String> {
    let content = line.strip_prefix("Format:").unwrap_or(line);
    content.split(',').map(|s| s.trim().to_string()).collect()
}

/// Parse style line.
fn parse_style_line(line: &str, format: &[String]) -> Option<(String, SubtitleStyle)> {
    let content = line.strip_prefix("Style:")?;
    let values: Vec<&str> = content.split(',').map(str::trim).collect();

    let mut style_map = HashMap::new();
    for (i, field) in format.iter().enumerate() {
        if let Some(&value) = values.get(i) {
            style_map.insert(field.clone(), value.to_string());
        }
    }

    let name = style_map.get("Name")?.clone();
    let mut style = SubtitleStyle::default();

    // Font name (ignored - we use the provided font)
    // Fontsize
    if let Some(size) = style_map.get("Fontsize") {
        if let Ok(size) = size.parse::<f32>() {
            style.font_size = size;
        }
    }

    // Primary color
    if let Some(color) = style_map.get("PrimaryColour") {
        if let Ok(c) = parse_ass_color(color) {
            style.primary_color = c;
        }
    }

    // Secondary color
    if let Some(color) = style_map.get("SecondaryColour") {
        if let Ok(c) = parse_ass_color(color) {
            style.secondary_color = c;
        }
    }

    // Outline color
    if let Some(color) = style_map.get("OutlineColour") {
        if let Ok(c) = parse_ass_color(color) {
            style.outline = Some(OutlineStyle::new(c, 2.0));
        }
    }

    // Outline width
    if let Some(width) = style_map.get("Outline") {
        if let Ok(w) = width.parse::<f32>() {
            if let Some(outline) = &mut style.outline {
                outline.width = w;
            }
        }
    }

    // Shadow
    if let Some(shadow) = style_map.get("Shadow") {
        if let Ok(s) = shadow.parse::<f32>() {
            if s > 0.0 {
                style.shadow = Some(ShadowStyle::new(Color::black(), s, s, 0.0));
            }
        }
    }

    // Alignment (SSA uses numeric alignment)
    if let Some(align) = style_map.get("Alignment") {
        if let Ok(a) = align.parse::<u8>() {
            style.alignment = parse_ass_alignment(a);
        }
    }

    // Margins
    if let Some(margin) = style_map.get("MarginL") {
        if let Ok(m) = margin.parse::<u32>() {
            style.margin_left = m;
        }
    }
    if let Some(margin) = style_map.get("MarginR") {
        if let Ok(m) = margin.parse::<u32>() {
            style.margin_right = m;
        }
    }
    if let Some(margin) = style_map.get("MarginV") {
        if let Ok(m) = margin.parse::<u32>() {
            style.margin_bottom = m;
            style.margin_top = m;
        }
    }

    Some((name, style))
}

/// Parse event/dialogue line.
fn parse_event_line(
    line: &str,
    format: &[String],
    styles: &HashMap<String, SubtitleStyle>,
) -> Option<Subtitle> {
    let is_comment = line.starts_with("Comment:");
    let content = line
        .strip_prefix("Dialogue:")
        .or_else(|| line.strip_prefix("Comment:"))?;

    // Split carefully - text field may contain commas
    let parts: Vec<&str> = content.splitn(format.len(), ',').map(str::trim).collect();

    let mut event_map = HashMap::new();
    for (i, field) in format.iter().enumerate() {
        if let Some(&value) = parts.get(i) {
            event_map.insert(field.clone(), value.to_string());
        }
    }

    // Skip comments
    if is_comment {
        return None;
    }

    // Parse start and end times
    let start_str = event_map.get("Start")?;
    let end_str = event_map.get("End")?;

    let start_time = parse_ass_timestamp(start_str)?;
    let end_time = parse_ass_timestamp(end_str)?;

    // Get text
    let text = event_map.get("Text")?.clone();
    let text = strip_ass_tags(&text);

    // Get style
    let style_name = event_map
        .get("Style")
        .map(String::as_str)
        .unwrap_or("Default");
    let style = styles.get(style_name).cloned();

    let mut subtitle = Subtitle::new(start_time, end_time, text);
    subtitle.style = style;

    Some(subtitle)
}

/// Parse ASS timestamp (e.g., "0:00:01.00").
fn parse_ass_timestamp(ts: &str) -> Option<i64> {
    let parts: Vec<&str> = ts.split(':').collect();
    if parts.len() != 3 {
        return None;
    }

    let hours: i64 = parts[0].parse().ok()?;
    let minutes: i64 = parts[1].parse().ok()?;

    let sec_parts: Vec<&str> = parts[2].split('.').collect();
    if sec_parts.len() != 2 {
        return None;
    }

    let seconds: i64 = sec_parts[0].parse().ok()?;
    let centiseconds: i64 = sec_parts[1].parse().ok()?;

    Some(hours * 3600000 + minutes * 60000 + seconds * 1000 + centiseconds * 10)
}

/// Parse ASS color (format: &HAABBGGRR or &HAABBGGRR&).
fn parse_ass_color(color: &str) -> Result<Color, SubtitleError> {
    let color = color.trim_start_matches('&').trim_start_matches('H');
    let color = color.trim_end_matches('&');

    // ASS colors are in AABBGGRR format (reversed from typical RRGGBBAA)
    if color.len() < 6 {
        return Err(SubtitleError::InvalidColor(color.to_string()));
    }

    // Pad to 8 characters if needed (default alpha is FF)
    let padded = if color.len() == 6 {
        format!("FF{color}")
    } else {
        color.to_string()
    };

    let aa = u8::from_str_radix(&padded[0..2], 16)
        .map_err(|_| SubtitleError::InvalidColor(color.to_string()))?;
    let bb = u8::from_str_radix(&padded[2..4], 16)
        .map_err(|_| SubtitleError::InvalidColor(color.to_string()))?;
    let gg = u8::from_str_radix(&padded[4..6], 16)
        .map_err(|_| SubtitleError::InvalidColor(color.to_string()))?;
    let rr = u8::from_str_radix(&padded[6..8], 16)
        .map_err(|_| SubtitleError::InvalidColor(color.to_string()))?;

    // ASS alpha is inverted (0 = opaque, 255 = transparent)
    let alpha = 255 - aa;

    Ok(Color::new(rr, gg, bb, alpha))
}

/// Parse ASS alignment (numeric).
fn parse_ass_alignment(align: u8) -> Alignment {
    // ASS alignment: 1-3 = bottom, 4-6 = middle, 7-9 = top
    // Within each row: 1,4,7 = left, 2,5,8 = center, 3,6,9 = right
    match align % 3 {
        1 => Alignment::Left,
        2 => Alignment::Center,
        0 => Alignment::Right,
        _ => Alignment::Center,
    }
}

/// Strip ASS override tags from text.
fn strip_ass_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_tag = false;
    let mut brace_depth = 0u32;

    for c in text.chars() {
        match c {
            '{' => {
                in_tag = true;
                brace_depth += 1;
            }
            '}' => {
                if brace_depth > 0 {
                    brace_depth -= 1;
                }
                if brace_depth == 0 {
                    in_tag = false;
                }
            }
            _ => {
                if !in_tag {
                    result.push(c);
                }
            }
        }
    }

    // Handle line breaks
    result.replace("\\N", "\n").replace("\\n", "\n")
}

// ============================================================================
// ASS override tag parser
// ============================================================================

/// Rectangular clip region (in script coordinate space).
#[derive(Clone, Debug, PartialEq)]
pub struct ClipRect {
    /// Left edge.
    pub x1: f32,
    /// Top edge.
    pub y1: f32,
    /// Right edge.
    pub x2: f32,
    /// Bottom edge.
    pub y2: f32,
    /// Whether this is an inverse clip (`\iclip`).
    pub inverse: bool,
}

/// Origin point for rotation (`\org`).
#[derive(Clone, Debug, PartialEq)]
pub struct Origin {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

/// Fade animation from `\fad(t1,t2)`.
#[derive(Clone, Debug, PartialEq)]
pub struct FadTiming {
    /// Fade-in duration in milliseconds.
    pub fade_in_ms: u32,
    /// Fade-out duration in milliseconds.
    pub fade_out_ms: u32,
}

/// Move animation from `\move(x1,y1,x2,y2[,t1,t2])`.
#[derive(Clone, Debug, PartialEq)]
pub struct MoveAnimation {
    /// Start X position.
    pub x1: f32,
    /// Start Y position.
    pub y1: f32,
    /// End X position.
    pub x2: f32,
    /// End Y position.
    pub y2: f32,
    /// Optional start time within the event (milliseconds).
    pub t1: Option<i64>,
    /// Optional end time within the event (milliseconds).
    pub t2: Option<i64>,
}

/// All recognized override tags extracted from a single `{...}` block.
#[derive(Clone, Debug, Default)]
pub struct AssOverrideTags {
    /// Clip region (from `\clip` or `\iclip`).
    pub clip: Option<ClipRect>,
    /// Rotation origin (from `\org`).
    pub origin: Option<Origin>,
    /// Fade timing (from `\fad`).
    pub fad: Option<FadTiming>,
    /// Move animation (from `\move`).
    pub movement: Option<MoveAnimation>,
    /// Remaining unrecognized tag text.
    pub unknown_tags: Vec<String>,
}

/// Parse all override tag blocks in an ASS text string.
///
/// Iterates over every `{...}` block and extracts recognized override tags.
///
/// # Example
///
/// ```
/// use oximedia_subtitle::parser::ssa::parse_override_tags;
/// let tags = parse_override_tags(r"{\clip(10,20,200,150)\fad(300,500)}Hello");
/// assert!(tags.clip.is_some());
/// assert!(tags.fad.is_some());
/// ```
#[must_use]
pub fn parse_override_tags(text: &str) -> AssOverrideTags {
    let mut result = AssOverrideTags::default();

    let mut remaining = text;
    while let Some(open) = remaining.find('{') {
        let after = &remaining[open + 1..];
        if let Some(close) = after.find('}') {
            let block = &after[..close];
            parse_override_block(block, &mut result);
            remaining = &after[close + 1..];
        } else {
            break;
        }
    }

    result
}

/// Parse a single override block content (without the braces).
fn parse_override_block(block: &str, out: &mut AssOverrideTags) {
    // Split on backslash to get individual tags
    for raw_tag in block.split('\\') {
        let tag = raw_tag.trim();
        if tag.is_empty() {
            continue;
        }

        if let Some(args_str) = tag.strip_prefix("iclip(").and_then(|s| s.strip_suffix(')')) {
            if let Some(rect) = parse_clip_args(args_str, true) {
                out.clip = Some(rect);
            }
        } else if let Some(args_str) = tag.strip_prefix("clip(").and_then(|s| s.strip_suffix(')')) {
            if let Some(rect) = parse_clip_args(args_str, false) {
                out.clip = Some(rect);
            }
        } else if let Some(args_str) = tag.strip_prefix("org(").and_then(|s| s.strip_suffix(')')) {
            if let Some(org) = parse_two_floats(args_str) {
                out.origin = Some(Origin { x: org.0, y: org.1 });
            }
        } else if let Some(args_str) = tag.strip_prefix("fad(").and_then(|s| s.strip_suffix(')')) {
            if let Some(fad) = parse_fad_args(args_str) {
                out.fad = Some(fad);
            }
        } else if let Some(args_str) = tag.strip_prefix("move(").and_then(|s| s.strip_suffix(')')) {
            if let Some(mv) = parse_move_args(args_str) {
                out.movement = Some(mv);
            }
        } else {
            // Store unrecognized tags for downstream use
            out.unknown_tags.push(tag.to_string());
        }
    }
}

/// Parse `\clip` / `\iclip` rectangular arguments: "x1,y1,x2,y2".
fn parse_clip_args(args: &str, inverse: bool) -> Option<ClipRect> {
    let parts: Vec<&str> = args.split(',').collect();
    if parts.len() == 4 {
        let x1 = parts[0].trim().parse::<f32>().ok()?;
        let y1 = parts[1].trim().parse::<f32>().ok()?;
        let x2 = parts[2].trim().parse::<f32>().ok()?;
        let y2 = parts[3].trim().parse::<f32>().ok()?;
        Some(ClipRect {
            x1,
            y1,
            x2,
            y2,
            inverse,
        })
    } else {
        None
    }
}

/// Parse two comma-separated floats.
fn parse_two_floats(args: &str) -> Option<(f32, f32)> {
    let mut it = args.split(',');
    let x = it.next()?.trim().parse::<f32>().ok()?;
    let y = it.next()?.trim().parse::<f32>().ok()?;
    Some((x, y))
}

/// Parse `\fad(t1,t2)` arguments.
fn parse_fad_args(args: &str) -> Option<FadTiming> {
    let mut it = args.split(',');
    let t1 = it.next()?.trim().parse::<u32>().ok()?;
    let t2 = it.next()?.trim().parse::<u32>().ok()?;
    Some(FadTiming {
        fade_in_ms: t1,
        fade_out_ms: t2,
    })
}

/// Parse `\move(x1,y1,x2,y2[,t1,t2])` arguments.
fn parse_move_args(args: &str) -> Option<MoveAnimation> {
    let parts: Vec<&str> = args.split(',').collect();
    if parts.len() < 4 {
        return None;
    }
    let x1 = parts[0].trim().parse::<f32>().ok()?;
    let y1 = parts[1].trim().parse::<f32>().ok()?;
    let x2 = parts[2].trim().parse::<f32>().ok()?;
    let y2 = parts[3].trim().parse::<f32>().ok()?;

    let (t1, t2) = if parts.len() >= 6 {
        let t1 = parts[4].trim().parse::<i64>().ok();
        let t2 = parts[5].trim().parse::<i64>().ok();
        (t1, t2)
    } else {
        (None, None)
    };

    Some(MoveAnimation {
        x1,
        y1,
        x2,
        y2,
        t1,
        t2,
    })
}

#[cfg(test)]
mod ass_override_tests {
    use super::*;

    #[test]
    fn test_parse_clip_basic() {
        let tags = parse_override_tags(r"{\clip(10,20,200,150)}Hello");
        assert!(tags.clip.is_some());
        let clip = tags.clip.expect("clip present");
        assert!((clip.x1 - 10.0).abs() < f32::EPSILON);
        assert!((clip.y1 - 20.0).abs() < f32::EPSILON);
        assert!((clip.x2 - 200.0).abs() < f32::EPSILON);
        assert!((clip.y2 - 150.0).abs() < f32::EPSILON);
        assert!(!clip.inverse);
    }

    #[test]
    fn test_parse_iclip() {
        let tags = parse_override_tags(r"{\iclip(5,10,100,200)}Text");
        let clip = tags.clip.expect("iclip present");
        assert!(clip.inverse);
        assert!((clip.x1 - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_org() {
        let tags = parse_override_tags(r"{\org(320,240)}Rotated");
        let org = tags.origin.expect("org present");
        assert!((org.x - 320.0).abs() < f32::EPSILON);
        assert!((org.y - 240.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_fad() {
        let tags = parse_override_tags(r"{\fad(300,500)}Fading");
        let fad = tags.fad.expect("fad present");
        assert_eq!(fad.fade_in_ms, 300);
        assert_eq!(fad.fade_out_ms, 500);
    }

    #[test]
    fn test_parse_move_without_time() {
        let tags = parse_override_tags(r"{\move(100,200,400,300)}Moving");
        let mv = tags.movement.expect("move present");
        assert!((mv.x1 - 100.0).abs() < f32::EPSILON);
        assert!((mv.y1 - 200.0).abs() < f32::EPSILON);
        assert!((mv.x2 - 400.0).abs() < f32::EPSILON);
        assert!((mv.y2 - 300.0).abs() < f32::EPSILON);
        assert!(mv.t1.is_none());
        assert!(mv.t2.is_none());
    }

    #[test]
    fn test_parse_move_with_time() {
        let tags = parse_override_tags(r"{\move(0,0,640,360,100,900)}Moving timed");
        let mv = tags.movement.expect("move timed present");
        assert_eq!(mv.t1, Some(100));
        assert_eq!(mv.t2, Some(900));
    }

    #[test]
    fn test_parse_combined_tags() {
        let tags = parse_override_tags(r"{\clip(0,0,320,240)\fad(200,300)\org(160,120)}Text");
        assert!(tags.clip.is_some());
        assert!(tags.fad.is_some());
        assert!(tags.origin.is_some());
    }

    #[test]
    fn test_strip_tags_with_override() {
        // The main parser should strip override tags leaving clean text
        let text = r"{\clip(0,0,100,100)\fad(100,100)}Hello World";
        let stripped = strip_ass_tags(text);
        assert_eq!(stripped.trim(), "Hello World");
    }

    #[test]
    fn test_parse_override_no_tags() {
        let tags = parse_override_tags("No override tags here");
        assert!(tags.clip.is_none());
        assert!(tags.fad.is_none());
        assert!(tags.origin.is_none());
        assert!(tags.movement.is_none());
    }

    #[test]
    fn test_parse_multiple_blocks() {
        let tags = parse_override_tags(r"{\clip(10,10,100,100)}Mid{\fad(50,50)}");
        // Should accumulate last seen value; clip from first block
        assert!(tags.clip.is_some());
        assert!(tags.fad.is_some());
    }

    #[test]
    fn test_parse_move_insufficient_args_returns_none() {
        let tags = parse_override_tags(r"{\move(100,200)}Bad move");
        assert!(tags.movement.is_none());
    }

    #[test]
    fn test_parse_fad_zero() {
        let tags = parse_override_tags(r"{\fad(0,0)}Instant");
        let fad = tags.fad.expect("fad zero");
        assert_eq!(fad.fade_in_ms, 0);
        assert_eq!(fad.fade_out_ms, 0);
    }

    #[test]
    fn test_clip_rect_fields() {
        let clip = ClipRect {
            x1: 1.0,
            y1: 2.0,
            x2: 3.0,
            y2: 4.0,
            inverse: true,
        };
        assert!((clip.x2 - 3.0).abs() < f32::EPSILON);
        assert!(clip.inverse);
    }

    #[test]
    fn test_fad_timing_fields() {
        let fad = FadTiming {
            fade_in_ms: 100,
            fade_out_ms: 200,
        };
        assert_eq!(fad.fade_in_ms, 100);
    }

    #[test]
    fn test_origin_fields() {
        let org = Origin { x: 1.5, y: 2.5 };
        assert!((org.x - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_move_animation_no_time() {
        let mv = MoveAnimation {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 100.0,
            t1: None,
            t2: None,
        };
        assert!(mv.t1.is_none());
    }
}

/// Format milliseconds as ASS timestamp.
#[must_use]
pub fn format_timestamp(ms: i64) -> String {
    let hours = ms / 3600000;
    let minutes = (ms % 3600000) / 60000;
    let seconds = (ms % 60000) / 1000;
    let centis = (ms % 1000) / 10;

    format!("{hours}:{minutes:02}:{seconds:02}.{centis:02}")
}

/// Write subtitles in ASS format.
///
/// # Errors
///
/// Returns error if writing fails.
pub fn write(subtitles: &[Subtitle]) -> SubtitleResult<String> {
    let mut output = String::new();

    // Script info
    output.push_str("[Script Info]\n");
    output.push_str("Title: Exported Subtitles\n");
    output.push_str("ScriptType: v4.00+\n");
    output.push_str("WrapStyle: 0\n");
    output.push_str("ScaledBorderAndShadow: yes\n");
    output.push_str("YCbCr Matrix: TV.709\n\n");

    // Default style
    output.push_str("[V4+ Styles]\n");
    output.push_str("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n");
    output.push_str("Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,40,40,40,1\n\n");

    // Events
    output.push_str("[Events]\n");
    output.push_str(
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
    );

    for subtitle in subtitles {
        let start = format_timestamp(subtitle.start_time);
        let end = format_timestamp(subtitle.end_time);
        let text = subtitle.text.replace('\n', "\\N");

        output.push_str(&format!(
            "Dialogue: 0,{},{},Default,,0,0,0,,{}\n",
            start, end, text
        ));
    }

    Ok(output)
}

// ============================================================================
// AssStyleCache wiring tests
// ============================================================================

#[cfg(test)]
mod style_cache_tests {
    use super::*;

    /// Minimal ASS document with two named styles and three dialogue lines.
    const SAMPLE_ASS: &str = r#"[Script Info]
Title: Test
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,40,40,40,1
Style: Title,Arial,64,&H0000FFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,8,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Hello world!
Dialogue: 0,0:00:05.00,0:00:08.00,Title,,0,0,0,,Title line
Dialogue: 0,0:00:09.00,0:00:12.00,Default,,0,0,0,,Another line
"#;

    #[test]
    fn test_ass_style_cache_wiring() {
        let file = parse_ass(SAMPLE_ASS).expect("parse should succeed");

        // The cache must have been built from the parsed styles.
        assert_eq!(
            file.style_cache.len(),
            2,
            "cache should contain both styles"
        );
        assert!(
            file.style_cache.contains("Default"),
            "Default must be in cache"
        );
        assert!(file.style_cache.contains("Title"), "Title must be in cache");

        // Verify that cache font sizes match the parsed styles map.
        let default_style = file.styles.get("Default").expect("Default in map");
        let title_style = file.styles.get("Title").expect("Title in map");
        let cached_default = file.style_cache.get("Default").expect("Default in cache");
        let cached_title = file.style_cache.get("Title").expect("Title in cache");

        assert!(
            (cached_default.font_size - default_style.font_size).abs() < 0.5,
            "cache Default font_size must match parsed value"
        );
        assert!(
            (cached_title.font_size - title_style.font_size).abs() < 0.5,
            "cache Title font_size must match parsed value"
        );

        // Verify the three dialogue lines received correct styles.
        assert_eq!(file.events.len(), 3);
        // Line 0 -> Default (font_size 48)
        let ev0_style = file.events[0].style.as_ref().expect("event 0 has style");
        assert!(
            (ev0_style.font_size - 48.0).abs() < 0.5,
            "event 0 should use Default (48px)"
        );
        // Line 1 -> Title (font_size 64)
        let ev1_style = file.events[1].style.as_ref().expect("event 1 has style");
        assert!(
            (ev1_style.font_size - 64.0).abs() < 0.5,
            "event 1 should use Title (64px)"
        );
        // Line 2 -> Default (font_size 48)
        let ev2_style = file.events[2].style.as_ref().expect("event 2 has style");
        assert!(
            (ev2_style.font_size - 48.0).abs() < 0.5,
            "event 2 should use Default (48px)"
        );
    }

    #[test]
    fn test_ass_cache_fallback_for_unknown_style() {
        // A dialogue line referencing a nonexistent style should not crash and
        // should produce a subtitle (style will be None because HashMap lookup
        // returns None for unknown names, which is the existing behavior).
        let ass = r#"[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.00,Ghost,,0,0,0,,Phantom line
"#;
        let file = parse_ass(ass).expect("parse should succeed");
        assert_eq!(file.events.len(), 1);
        // cache was built from styles (only "Default" registered)
        assert!(file.style_cache.contains("Default"));
        assert!(!file.style_cache.contains("Ghost"));
    }

    #[test]
    fn test_ass_cache_and_direct_lookup_agree() {
        // Verify that cache.get(name) == styles.get(name) for each registered name.
        let file = parse_ass(SAMPLE_ASS).expect("parse should succeed");
        for (name, style_from_map) in &file.styles {
            let from_cache = file
                .style_cache
                .get(name)
                .expect("every style in map must be in cache");
            assert!(
                (from_cache.font_size - style_from_map.font_size).abs() < 0.5,
                "cache and map must agree on font_size for style {name}"
            );
        }
    }
}
