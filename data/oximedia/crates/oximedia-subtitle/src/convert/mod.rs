//! Subtitle format conversion to CEA-608/708.
//!
//! This module provides conversion functions from text-based subtitle formats
//! (SRT, WebVTT, SSA/ASS) to CEA-608 and CEA-708 closed captions.

use crate::cea::{
    Cea608Channel, Cea608Color, Cea608Encoder, Cea608Mode, Cea708Color, Cea708Encoder,
    Cea708FontStyle, Cea708Opacity, Cea708PenAttributes, Cea708PenColor, Cea708PenSize,
    Cea708ServiceNumber, Cea708WindowAttributes, Cea708WindowId, FrameRate, FrameRateAdapter,
};
use crate::style::{Alignment, Color};
use crate::{Subtitle, SubtitleError, SubtitleResult};

/// Conversion options for CEA encoding.
#[derive(Clone, Debug)]
pub struct CeaConversionOptions {
    /// Target frame rate.
    pub frame_rate: FrameRate,
    /// CEA-608 channel (if converting to 608).
    pub cea608_channel: Cea608Channel,
    /// CEA-608 display mode.
    pub cea608_mode: Cea608Mode,
    /// CEA-708 service number (if converting to 708).
    pub cea708_service: Cea708ServiceNumber,
    /// Maximum characters per line.
    pub max_chars_per_line: usize,
    /// Maximum number of lines.
    pub max_lines: usize,
    /// Enable automatic line breaking.
    pub auto_line_break: bool,
    /// Strip formatting tags.
    pub strip_formatting: bool,
}

impl Default for CeaConversionOptions {
    fn default() -> Self {
        Self {
            frame_rate: FrameRate::ntsc(),
            cea608_channel: Cea608Channel::CC1,
            cea608_mode: Cea608Mode::PopOn,
            cea708_service: Cea708ServiceNumber::new(1).expect("hardcoded value is valid"),
            max_chars_per_line: 32,
            max_lines: 4,
            auto_line_break: true,
            strip_formatting: true,
        }
    }
}

/// CEA-608 output data with timing.
#[derive(Clone, Debug)]
pub struct Cea608Output {
    /// Start time in milliseconds.
    pub start_time: i64,
    /// End time in milliseconds.
    pub end_time: i64,
    /// Encoded byte pairs.
    pub data: Vec<(u8, u8)>,
}

/// CEA-708 output data with timing.
#[derive(Clone, Debug)]
pub struct Cea708Output {
    /// Start time in milliseconds.
    pub start_time: i64,
    /// End time in milliseconds.
    pub end_time: i64,
    /// Service block data.
    pub service_block: Vec<u8>,
    /// CDP (Caption Distribution Packet) data.
    pub cdp: Vec<u8>,
}

/// Convert SRT subtitles to CEA-608.
///
/// # Errors
///
/// Returns error if conversion fails.
pub fn srt_to_cea608(
    subtitles: &[Subtitle],
    options: &CeaConversionOptions,
) -> SubtitleResult<Vec<Cea608Output>> {
    let mut output = Vec::new();
    let mut encoder = Cea608Encoder::new(options.cea608_channel);

    encoder.set_mode(options.cea608_mode);

    for subtitle in subtitles {
        // Clear previous caption
        encoder.clear_buffer();

        // Process text
        let text = if options.strip_formatting {
            strip_html_tags(&subtitle.text)
        } else {
            subtitle.text.clone()
        };

        // Split into lines
        let lines = split_into_lines(&text, options.max_chars_per_line, options.max_lines);

        // Encode each line
        for (row, line) in lines.iter().enumerate() {
            let row_num = 15 - (lines.len() - 1) as u8 + row as u8;
            encoder.set_position(row_num, 0);
            encoder.add_text(line)?;
        }

        // End caption (for pop-on mode)
        if matches!(options.cea608_mode, Cea608Mode::PopOn) {
            encoder.end_caption();
        }

        // Get output
        let data = encoder.take_output();

        output.push(Cea608Output {
            start_time: subtitle.start_time,
            end_time: subtitle.end_time,
            data,
        });
    }

    Ok(output)
}

/// Convert WebVTT subtitles to CEA-608.
///
/// # Errors
///
/// Returns error if conversion fails.
pub fn webvtt_to_cea608(
    subtitles: &[Subtitle],
    options: &CeaConversionOptions,
) -> SubtitleResult<Vec<Cea608Output>> {
    // WebVTT is similar to SRT but may have positioning cues
    let mut output = Vec::new();
    let mut encoder = Cea608Encoder::new(options.cea608_channel);

    encoder.set_mode(options.cea608_mode);

    for subtitle in subtitles {
        encoder.clear_buffer();

        // Strip WebVTT tags and cue settings
        let text = strip_webvtt_tags(&subtitle.text);

        // Convert positioning hints to row selection
        let row = if let Some(pos) = &subtitle.position {
            map_position_to_row(pos.y)
        } else {
            15 // Default to bottom
        };

        let lines = split_into_lines(&text, options.max_chars_per_line, options.max_lines);

        for (line_offset, line) in lines.iter().enumerate() {
            encoder.set_position(row - line_offset as u8, 0);
            encoder.add_text(line)?;
        }

        if matches!(options.cea608_mode, Cea608Mode::PopOn) {
            encoder.end_caption();
        }

        output.push(Cea608Output {
            start_time: subtitle.start_time,
            end_time: subtitle.end_time,
            data: encoder.take_output(),
        });
    }

    Ok(output)
}

/// Convert SSA/ASS subtitles to CEA-608.
///
/// # Errors
///
/// Returns error if conversion fails.
pub fn ssa_to_cea608(
    subtitles: &[Subtitle],
    options: &CeaConversionOptions,
) -> SubtitleResult<Vec<Cea608Output>> {
    let mut output = Vec::new();
    let mut encoder = Cea608Encoder::new(options.cea608_channel);

    encoder.set_mode(options.cea608_mode);

    for subtitle in subtitles {
        encoder.clear_buffer();

        // Parse SSA/ASS style tags
        let (text, color) = parse_ssa_style(&subtitle.text);

        // Set color if available
        if let Some(cea_color) = map_color_to_cea608(color) {
            encoder.set_style(cea_color, false, false);
        }

        let lines = split_into_lines(&text, options.max_chars_per_line, options.max_lines);

        for (row, line) in lines.iter().enumerate() {
            let row_num = 15 - (lines.len() - 1) as u8 + row as u8;
            encoder.set_position(row_num, 0);
            encoder.add_text(line)?;
        }

        if matches!(options.cea608_mode, Cea608Mode::PopOn) {
            encoder.end_caption();
        }

        output.push(Cea608Output {
            start_time: subtitle.start_time,
            end_time: subtitle.end_time,
            data: encoder.take_output(),
        });
    }

    Ok(output)
}

/// Convert SRT subtitles to CEA-708.
///
/// # Errors
///
/// Returns error if conversion fails.
pub fn srt_to_cea708(
    subtitles: &[Subtitle],
    options: &CeaConversionOptions,
) -> SubtitleResult<Vec<Cea708Output>> {
    let mut output = Vec::new();
    let mut encoder = Cea708Encoder::new(options.cea708_service);

    // Define default window
    let window_id = Cea708WindowId::new(0)?;
    let window_attrs = Cea708WindowAttributes::default();
    encoder.define_window(window_id, window_attrs)?;
    encoder.set_current_window(window_id);

    for subtitle in subtitles {
        // Clear window
        encoder.clear_windows(0x01);

        // Process text
        let text = if options.strip_formatting {
            strip_html_tags(&subtitle.text)
        } else {
            subtitle.text.clone()
        };

        // Add text
        encoder.add_text(&text)?;

        // Display window
        encoder.display_windows(0x01);

        // Build output
        let service_block = encoder.build_service_block();
        let framerate_code = crate::cea::get_framerate_code(options.frame_rate.as_float());
        let cdp = encoder.build_cdp(framerate_code, None);

        output.push(Cea708Output {
            start_time: subtitle.start_time,
            end_time: subtitle.end_time,
            service_block,
            cdp,
        });
    }

    Ok(output)
}

/// Convert WebVTT subtitles to CEA-708.
///
/// # Errors
///
/// Returns error if conversion fails.
pub fn webvtt_to_cea708(
    subtitles: &[Subtitle],
    options: &CeaConversionOptions,
) -> SubtitleResult<Vec<Cea708Output>> {
    let mut output = Vec::new();
    let mut encoder = Cea708Encoder::new(options.cea708_service);

    let window_id = Cea708WindowId::new(0)?;
    let mut window_attrs = Cea708WindowAttributes::default();

    for subtitle in subtitles {
        // Update window position based on WebVTT cue settings
        if let Some(pos) = &subtitle.position {
            window_attrs.anchor.vertical = (pos.y * 100.0) as u8;
            window_attrs.anchor.horizontal = (pos.x * 100.0) as u8;
        }

        encoder.define_window(window_id, window_attrs)?;
        encoder.set_current_window(window_id);

        // Clear and add text
        encoder.clear_windows(0x01);

        let text = strip_webvtt_tags(&subtitle.text);
        encoder.add_text(&text)?;

        encoder.display_windows(0x01);

        let service_block = encoder.build_service_block();
        let framerate_code = crate::cea::get_framerate_code(options.frame_rate.as_float());
        let cdp = encoder.build_cdp(framerate_code, None);

        output.push(Cea708Output {
            start_time: subtitle.start_time,
            end_time: subtitle.end_time,
            service_block,
            cdp,
        });
    }

    Ok(output)
}

/// Convert SSA/ASS subtitles to CEA-708.
///
/// # Errors
///
/// Returns error if conversion fails.
pub fn ssa_to_cea708(
    subtitles: &[Subtitle],
    options: &CeaConversionOptions,
) -> SubtitleResult<Vec<Cea708Output>> {
    let mut output = Vec::new();
    let mut encoder = Cea708Encoder::new(options.cea708_service);

    let window_id = Cea708WindowId::new(0)?;
    let window_attrs = Cea708WindowAttributes::default();
    encoder.define_window(window_id, window_attrs)?;

    for subtitle in subtitles {
        encoder.set_current_window(window_id);
        encoder.clear_windows(0x01);

        // Parse SSA/ASS style
        let (text, color) = parse_ssa_style(&subtitle.text);

        // Set pen color based on style
        if let Some(cea_color) = map_color_to_cea708(color) {
            let pen_color = Cea708PenColor {
                foreground: cea_color,
                ..Default::default()
            };
            encoder.set_pen_color(pen_color);
        }

        encoder.add_text(&text)?;
        encoder.display_windows(0x01);

        let service_block = encoder.build_service_block();
        let framerate_code = crate::cea::get_framerate_code(options.frame_rate.as_float());
        let cdp = encoder.build_cdp(framerate_code, None);

        output.push(Cea708Output {
            start_time: subtitle.start_time,
            end_time: subtitle.end_time,
            service_block,
            cdp,
        });
    }

    Ok(output)
}

/// Batch converter for multiple subtitle formats.
pub struct SubtitleConverter {
    options: CeaConversionOptions,
}

impl SubtitleConverter {
    /// Create a new subtitle converter.
    #[must_use]
    pub const fn new(options: CeaConversionOptions) -> Self {
        Self { options }
    }

    /// Convert to CEA-608 (auto-detect format).
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn to_cea608(&self, subtitles: &[Subtitle]) -> SubtitleResult<Vec<Cea608Output>> {
        // Use SRT conversion as default
        srt_to_cea608(subtitles, &self.options)
    }

    /// Convert to CEA-708 (auto-detect format).
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn to_cea708(&self, subtitles: &[Subtitle]) -> SubtitleResult<Vec<Cea708Output>> {
        srt_to_cea708(subtitles, &self.options)
    }

    /// Get the conversion options.
    #[must_use]
    pub const fn options(&self) -> &CeaConversionOptions {
        &self.options
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Strip HTML tags from text (SRT).
fn strip_html_tags(text: &str) -> String {
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

    result
}

/// Strip WebVTT tags and cue settings.
fn strip_webvtt_tags(text: &str) -> String {
    // Remove WebVTT voice tags <v Name>
    let mut result = text
        .replace("<v ", "")
        .replace("</v>", "")
        .replace("<c>", "")
        .replace("</c>", "");

    // Remove timestamp tags
    if let Some(idx) = result.find('<') {
        if let Some(end_idx) = result[idx..].find('>') {
            result = result[..idx].to_string() + &result[idx + end_idx + 1..];
        }
    }

    result
}

/// Parse SSA/ASS style tags.
fn parse_ssa_style(text: &str) -> (String, Option<Color>) {
    let mut clean_text = String::with_capacity(text.len());
    let mut color = None;

    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '{' {
            // Parse style override
            let mut tag = String::new();
            while let Some(&next_c) = chars.peek() {
                if next_c == '}' {
                    chars.next();
                    break;
                }
                tag.push(chars.next().expect("invariant: peek confirmed Some"));
            }

            // Parse color tag (\c&HBBGGRR&)
            if tag.starts_with("\\c&H") {
                if let Some(hex) = tag.strip_prefix("\\c&H") {
                    if let Some(hex_color) = hex.strip_suffix('&') {
                        // Parse BGR color
                        if let Ok(bgr) = u32::from_str_radix(hex_color, 16) {
                            let b = ((bgr >> 16) & 0xFF) as u8;
                            let g = ((bgr >> 8) & 0xFF) as u8;
                            let r = (bgr & 0xFF) as u8;
                            color = Some(Color::rgb(r, g, b));
                        }
                    }
                }
            }
        } else if c == '\\' && chars.peek() == Some(&'N') {
            // Line break
            chars.next();
            clean_text.push('\n');
        } else if c == '\\' && chars.peek() == Some(&'n') {
            // Soft line break
            chars.next();
            clean_text.push(' ');
        } else {
            clean_text.push(c);
        }
    }

    (clean_text, color)
}

/// Map color to CEA-608 color.
fn map_color_to_cea608(color: Option<Color>) -> Option<Cea608Color> {
    color.map(|c| {
        // Simple color matching
        if c.r > 200 && c.g > 200 && c.b > 200 {
            Cea608Color::White
        } else if c.r > 200 && c.g < 100 && c.b < 100 {
            Cea608Color::Red
        } else if c.r < 100 && c.g > 200 && c.b < 100 {
            Cea608Color::Green
        } else if c.r < 100 && c.g < 100 && c.b > 200 {
            Cea608Color::Blue
        } else if c.r > 200 && c.g > 200 && c.b < 100 {
            Cea608Color::Yellow
        } else if c.r > 200 && c.g < 100 && c.b > 200 {
            Cea608Color::Magenta
        } else if c.r < 100 && c.g > 200 && c.b > 200 {
            Cea608Color::Cyan
        } else {
            Cea608Color::White
        }
    })
}

/// Map color to CEA-708 color (2-bit per channel).
fn map_color_to_cea708(color: Option<Color>) -> Option<Cea708Color> {
    color.map(|c| {
        // Convert 8-bit to 2-bit color
        let r = (c.r >> 6) & 0x03;
        let g = (c.g >> 6) & 0x03;
        let b = (c.b >> 6) & 0x03;
        Cea708Color::new(r, g, b)
    })
}

/// Map vertical position (0.0-1.0) to CEA-608 row (1-15).
fn map_position_to_row(y: f32) -> u8 {
    // 0.0 = top (row 1), 1.0 = bottom (row 15)
    let row = (y * 14.0 + 1.0).round() as u8;
    row.clamp(1, 15)
}

/// Split text into lines with maximum length.
fn split_into_lines(text: &str, max_chars: usize, max_lines: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        if current_line.is_empty() {
            current_line = word.to_string();
        } else if current_line.len() + 1 + word.len() <= max_chars {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            // Start new line
            if lines.len() < max_lines {
                lines.push(current_line);
                current_line = word.to_string();
            } else {
                // Truncate if exceeds max lines
                break;
            }
        }
    }

    if !current_line.is_empty() && lines.len() < max_lines {
        lines.push(current_line);
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

/// Timing adjustment for caption display duration.
pub struct TimingAdjuster {
    min_duration_ms: i64,
    max_duration_ms: i64,
    gap_ms: i64,
}

impl TimingAdjuster {
    /// Create a new timing adjuster.
    #[must_use]
    pub const fn new(min_duration_ms: i64, max_duration_ms: i64, gap_ms: i64) -> Self {
        Self {
            min_duration_ms,
            max_duration_ms,
            gap_ms,
        }
    }

    /// Default timing adjuster.
    #[must_use]
    pub const fn default_adjuster() -> Self {
        Self::new(1000, 7000, 100)
    }

    /// Adjust subtitle timing.
    pub fn adjust(&self, subtitle: &mut Subtitle) {
        let duration = subtitle.end_time - subtitle.start_time;

        // Enforce minimum duration
        if duration < self.min_duration_ms {
            subtitle.end_time = subtitle.start_time + self.min_duration_ms;
        }

        // Enforce maximum duration
        if duration > self.max_duration_ms {
            subtitle.end_time = subtitle.start_time + self.max_duration_ms;
        }
    }

    /// Adjust timing for a list of subtitles (add gaps).
    pub fn adjust_list(&self, subtitles: &mut [Subtitle]) {
        for i in 0..subtitles.len() {
            self.adjust(&mut subtitles[i]);

            // Add gap to next subtitle
            if i + 1 < subtitles.len() {
                let end_time = subtitles[i].end_time;
                let next_start = subtitles[i + 1].start_time;

                if next_start < end_time + self.gap_ms {
                    // Adjust to maintain gap
                    subtitles[i].end_time = (next_start - self.gap_ms).max(subtitles[i].start_time);
                }
            }
        }
    }
}

/// Universal subtitle format converter.
///
/// Converts between all supported subtitle formats.
pub struct FormatConverter {
    /// Preserve style information when possible.
    pub preserve_styles: bool,
    /// Preserve timing information.
    pub preserve_timing: bool,
}

impl FormatConverter {
    /// Create a new format converter.
    #[must_use]
    pub const fn new(preserve_styles: bool, preserve_timing: bool) -> Self {
        Self {
            preserve_styles,
            preserve_timing,
        }
    }

    /// Convert SRT to WebVTT.
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn srt_to_webvtt(&self, subtitles: &[Subtitle]) -> SubtitleResult<String> {
        crate::parser::webvtt::write(subtitles)
    }

    /// Convert SRT to ASS.
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn srt_to_ass(&self, subtitles: &[Subtitle]) -> SubtitleResult<String> {
        crate::parser::ssa::write(subtitles)
    }

    /// Convert SRT to TTML.
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn srt_to_ttml(&self, subtitles: &[Subtitle]) -> SubtitleResult<String> {
        crate::parser::ttml::write(subtitles)
    }

    /// Convert WebVTT to SRT.
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn webvtt_to_srt(&self, subtitles: &[Subtitle]) -> SubtitleResult<String> {
        crate::parser::srt::write(subtitles)
    }

    /// Convert WebVTT to ASS.
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn webvtt_to_ass(&self, subtitles: &[Subtitle]) -> SubtitleResult<String> {
        crate::parser::ssa::write(subtitles)
    }

    /// Convert WebVTT to TTML.
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn webvtt_to_ttml(&self, subtitles: &[Subtitle]) -> SubtitleResult<String> {
        crate::parser::ttml::write(subtitles)
    }

    /// Convert ASS to SRT (styles are lost).
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn ass_to_srt(&self, subtitles: &[Subtitle]) -> SubtitleResult<String> {
        crate::parser::srt::write(subtitles)
    }

    /// Convert ASS to WebVTT.
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn ass_to_webvtt(&self, subtitles: &[Subtitle]) -> SubtitleResult<String> {
        crate::parser::webvtt::write(subtitles)
    }

    /// Convert ASS to TTML.
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn ass_to_ttml(&self, subtitles: &[Subtitle]) -> SubtitleResult<String> {
        crate::parser::ttml::write(subtitles)
    }

    /// Convert TTML to SRT.
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn ttml_to_srt(&self, subtitles: &[Subtitle]) -> SubtitleResult<String> {
        crate::parser::srt::write(subtitles)
    }

    /// Convert TTML to WebVTT.
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn ttml_to_webvtt(&self, subtitles: &[Subtitle]) -> SubtitleResult<String> {
        crate::parser::webvtt::write(subtitles)
    }

    /// Convert TTML to ASS.
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails.
    pub fn ttml_to_ass(&self, subtitles: &[Subtitle]) -> SubtitleResult<String> {
        crate::parser::ssa::write(subtitles)
    }
}

impl Default for FormatConverter {
    fn default() -> Self {
        Self::new(true, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_tags() {
        let text = "<b>Hello</b> <i>World</i>";
        assert_eq!(strip_html_tags(text), "Hello World");
    }

    #[test]
    fn test_split_lines() {
        let text = "This is a long line that should be split into multiple lines";
        let lines = split_into_lines(text, 20, 4);
        assert!(lines.len() > 1);
        assert!(lines.iter().all(|l| l.len() <= 20));
    }

    #[test]
    fn test_color_mapping() {
        let red = Color::rgb(255, 0, 0);
        assert_eq!(map_color_to_cea608(Some(red)), Some(Cea608Color::Red));

        let white = Color::rgb(255, 255, 255);
        assert_eq!(map_color_to_cea608(Some(white)), Some(Cea608Color::White));
    }

    #[test]
    fn test_position_mapping() {
        assert_eq!(map_position_to_row(0.0), 1);
        assert_eq!(map_position_to_row(1.0), 15);
        assert_eq!(map_position_to_row(0.5), 8);
    }

    #[test]
    fn test_timing_adjuster() {
        let mut sub = Subtitle::new(0, 500, "Test".to_string());
        let adjuster = TimingAdjuster::default_adjuster();
        adjuster.adjust(&mut sub);
        assert_eq!(sub.end_time - sub.start_time, 1000); // Minimum duration applied
    }

    #[test]
    fn test_format_converter() {
        let converter = FormatConverter::default();
        let subtitles = vec![Subtitle::new(1000, 2000, "Test".to_string())];

        // Test SRT to WebVTT
        let webvtt = converter
            .srt_to_webvtt(&subtitles)
            .expect("should succeed in test");
        assert!(webvtt.contains("WEBVTT"));

        // Test SRT to ASS
        let ass = converter
            .srt_to_ass(&subtitles)
            .expect("should succeed in test");
        assert!(ass.contains("[Script Info]"));

        // Test SRT to TTML
        let ttml = converter
            .srt_to_ttml(&subtitles)
            .expect("should succeed in test");
        assert!(ttml.contains("<tt"));
    }
}
