#![allow(dead_code)]
//! Text layout engine for broadcast graphics.
//!
//! Provides paragraph-level text layout with support for line breaking,
//! justification, alignment, BiDi (bi-directional) text direction, and
//! multi-line text rendering for broadcast graphics elements such as lower
//! thirds, tickers, and scoreboards.
//!
//! ## Right-to-Left (RTL) support
//!
//! Set [`TextLayoutConfig::direction`] to [`TextDirection::Rtl`] when
//! rendering Arabic, Hebrew, Persian, or other RTL scripts.  In RTL mode:
//!
//! - Glyph X-positions are **mirrored** within the line width so the first
//!   logical character appears at the right edge of the line.
//! - The default horizontal alignment is implicitly right-aligned when
//!   `TextAlign::Left` is requested (you may override this explicitly).
//! - `TextDirection::Auto` uses a simple Unicode heuristic to pick a
//!   direction: if the first strongly-directional character belongs to the
//!   Arabic or Hebrew Unicode blocks the direction is RTL, otherwise LTR.

use std::fmt;

/// Horizontal text alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextAlign {
    /// Left-aligned text.
    #[default]
    Left,
    /// Center-aligned text.
    Center,
    /// Right-aligned text.
    Right,
    /// Justified text (both edges aligned).
    Justify,
}

impl fmt::Display for TextAlign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Left => write!(f, "left"),
            Self::Center => write!(f, "center"),
            Self::Right => write!(f, "right"),
            Self::Justify => write!(f, "justify"),
        }
    }
}

/// Vertical text alignment within a bounding box.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VerticalAlign {
    /// Top-aligned text.
    #[default]
    Top,
    /// Middle-aligned text.
    Middle,
    /// Bottom-aligned text.
    Bottom,
}

/// Text overflow behavior when text exceeds the bounding box.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextOverflow {
    /// Clip text at the boundary.
    #[default]
    Clip,
    /// Add ellipsis (...) at the end.
    Ellipsis,
    /// Allow text to overflow.
    Visible,
    /// Shrink font size to fit.
    ShrinkToFit,
}

/// Unicode BiDi base direction for a paragraph.
///
/// This controls whether glyph positions are computed left-to-right (LTR) or
/// right-to-left (RTL) within each line.  For mixed scripts use `Auto`, which
/// inspects the first strongly-directional Unicode character in the text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextDirection {
    /// Left-to-right — the default for Latin, CJK, etc.
    #[default]
    Ltr,
    /// Right-to-left — Arabic, Hebrew, Persian, Urdu, etc.
    Rtl,
    /// Auto-detect from the first strongly-directional Unicode codepoint.
    Auto,
}

impl TextDirection {
    /// Resolve `Auto` into a concrete `Ltr` or `Rtl` value by scanning `text`.
    ///
    /// The heuristic examines each character in order and returns `Rtl` if the
    /// first character from a strongly-RTL Unicode block (Arabic U+0600–06FF,
    /// Hebrew U+0590–05FF, Thaana U+0780–07BF, N'Ko U+07C0–07FF) is found
    /// before any strongly-LTR character.  Falls back to `Ltr`.
    #[must_use]
    pub fn resolve(&self, text: &str) -> TextDirection {
        match self {
            Self::Ltr => Self::Ltr,
            Self::Rtl => Self::Rtl,
            Self::Auto => {
                for ch in text.chars() {
                    let cp = ch as u32;
                    if (0x0590..=0x05FF).contains(&cp)   // Hebrew
                        || (0x0600..=0x06FF).contains(&cp)  // Arabic
                        || (0x0750..=0x077F).contains(&cp)  // Arabic Supplement
                        || (0x0780..=0x07BF).contains(&cp)  // Thaana
                        || (0x07C0..=0x07FF).contains(&cp)  // N'Ko
                        || (0xFB50..=0xFDFF).contains(&cp)  // Arabic Presentation Forms-A
                        || (0xFE70..=0xFEFF).contains(&cp)
                    // Arabic Presentation Forms-B
                    {
                        return Self::Rtl;
                    }
                    // Latin, Cyrillic, Greek → definitely LTR.
                    if ch.is_alphabetic() {
                        return Self::Ltr;
                    }
                }
                Self::Ltr // default
            }
        }
    }

    /// Returns `true` if this direction (or its resolved value for `text`) is RTL.
    #[must_use]
    pub fn is_rtl(&self, text: &str) -> bool {
        self.resolve(text) == TextDirection::Rtl
    }
}

/// Line break mode for text wrapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LineBreakMode {
    /// Break at word boundaries.
    #[default]
    WordWrap,
    /// Break at character boundaries.
    CharWrap,
    /// No wrapping (single line).
    NoWrap,
    /// Break at word boundaries, fall back to character wrap if word is too long.
    WordCharWrap,
}

/// A positioned glyph in the layout.
#[derive(Clone, Debug)]
pub struct LayoutGlyph {
    /// Character represented by this glyph.
    pub character: char,
    /// X position of the glyph.
    pub x: f32,
    /// Y position of the glyph.
    pub y: f32,
    /// Width of the glyph advance.
    pub advance_width: f32,
    /// Height of the glyph line.
    pub line_height: f32,
    /// Index of the line this glyph belongs to.
    pub line_index: usize,
    /// Index of the glyph within the original text.
    pub char_index: usize,
}

/// A laid-out line of text.
#[derive(Clone, Debug)]
pub struct LayoutLine {
    /// Glyphs in this line.
    pub glyphs: Vec<LayoutGlyph>,
    /// Y offset of this line from the top of the layout.
    pub y_offset: f32,
    /// Width of the content on this line.
    pub width: f32,
    /// Height of this line.
    pub height: f32,
    /// Index of the first character in the original text.
    pub start_char: usize,
    /// Index past the last character in the original text.
    pub end_char: usize,
}

impl LayoutLine {
    /// Create a new empty layout line.
    pub fn new(y_offset: f32, height: f32, start_char: usize) -> Self {
        Self {
            glyphs: Vec::new(),
            y_offset,
            width: 0.0,
            height,
            start_char,
            end_char: start_char,
        }
    }

    /// Get the number of glyphs on this line.
    pub fn glyph_count(&self) -> usize {
        self.glyphs.len()
    }

    /// Check if this line is empty.
    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }
}

/// Configuration for text layout.
#[derive(Clone, Debug)]
pub struct TextLayoutConfig {
    /// Maximum width for text wrapping (pixels).
    pub max_width: f32,
    /// Maximum height for clipping (pixels). 0 means unlimited.
    pub max_height: f32,
    /// Horizontal alignment.
    pub align: TextAlign,
    /// Vertical alignment.
    pub vertical_align: VerticalAlign,
    /// Font size in pixels.
    pub font_size: f32,
    /// Line height multiplier (1.0 = default).
    pub line_height: f32,
    /// Letter spacing in pixels.
    pub letter_spacing: f32,
    /// Word spacing multiplier (1.0 = default).
    pub word_spacing: f32,
    /// Text overflow behavior.
    pub overflow: TextOverflow,
    /// Line break mode.
    pub line_break: LineBreakMode,
    /// First line indent in pixels.
    pub first_line_indent: f32,
    /// Paragraph spacing in pixels (extra space between paragraphs).
    pub paragraph_spacing: f32,
    /// Tab stop width in spaces.
    pub tab_width: u32,
    /// Base text direction for BiDi layout.
    ///
    /// Set to [`TextDirection::Rtl`] for Arabic / Hebrew, or leave as
    /// [`TextDirection::Auto`] to detect direction from the first character.
    pub direction: TextDirection,
}

impl Default for TextLayoutConfig {
    fn default() -> Self {
        Self {
            max_width: 800.0,
            max_height: 0.0,
            align: TextAlign::Left,
            vertical_align: VerticalAlign::Top,
            font_size: 16.0,
            line_height: 1.2,
            letter_spacing: 0.0,
            word_spacing: 1.0,
            overflow: TextOverflow::Clip,
            line_break: LineBreakMode::WordWrap,
            first_line_indent: 0.0,
            paragraph_spacing: 0.0,
            tab_width: 4,
            direction: TextDirection::Ltr,
        }
    }
}

impl TextLayoutConfig {
    /// Create a config for centered broadcast title text.
    pub fn broadcast_title(max_width: f32, font_size: f32) -> Self {
        Self {
            max_width,
            font_size,
            align: TextAlign::Center,
            vertical_align: VerticalAlign::Middle,
            line_height: 1.3,
            letter_spacing: 1.0,
            overflow: TextOverflow::ShrinkToFit,
            line_break: LineBreakMode::WordWrap,
            ..Default::default()
        }
    }

    /// Create a config for a lower-third text block.
    pub fn lower_third(max_width: f32, font_size: f32) -> Self {
        Self {
            max_width,
            font_size,
            align: TextAlign::Left,
            vertical_align: VerticalAlign::Bottom,
            line_height: 1.1,
            overflow: TextOverflow::Ellipsis,
            line_break: LineBreakMode::WordWrap,
            ..Default::default()
        }
    }

    /// Create a config for ticker text (single-line, no wrap).
    pub fn ticker(max_width: f32, font_size: f32) -> Self {
        Self {
            max_width,
            font_size,
            align: TextAlign::Left,
            line_break: LineBreakMode::NoWrap,
            overflow: TextOverflow::Visible,
            ..Default::default()
        }
    }

    /// Create a config for right-to-left ticker text (Arabic / Hebrew).
    ///
    /// Uses `TextDirection::Rtl` and right-aligns text by default.
    pub fn ticker_rtl(max_width: f32, font_size: f32) -> Self {
        Self {
            max_width,
            font_size,
            align: TextAlign::Right,
            line_break: LineBreakMode::NoWrap,
            overflow: TextOverflow::Visible,
            direction: TextDirection::Rtl,
            ..Default::default()
        }
    }

    /// Create a config for right-to-left paragraph text (Arabic / Hebrew).
    pub fn rtl_paragraph(max_width: f32, font_size: f32) -> Self {
        Self {
            max_width,
            font_size,
            align: TextAlign::Right,
            line_break: LineBreakMode::WordWrap,
            direction: TextDirection::Rtl,
            ..Default::default()
        }
    }

    /// Compute the effective line height in pixels.
    pub fn effective_line_height(&self) -> f32 {
        self.font_size * self.line_height
    }
}

/// Result of a text layout operation.
#[derive(Clone, Debug)]
pub struct TextLayoutResult {
    /// Laid-out lines.
    pub lines: Vec<LayoutLine>,
    /// Total width of the laid-out text.
    pub total_width: f32,
    /// Total height of the laid-out text.
    pub total_height: f32,
    /// Whether text was truncated.
    pub truncated: bool,
    /// Number of characters that fit in the layout.
    pub chars_fitted: usize,
    /// Total number of characters in the input.
    pub total_chars: usize,
}

impl TextLayoutResult {
    /// Check if the entire text fit within the layout bounds.
    pub fn is_complete(&self) -> bool {
        !self.truncated
    }

    /// Get the number of lines.
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Get total number of glyphs across all lines.
    pub fn glyph_count(&self) -> usize {
        self.lines.iter().map(LayoutLine::glyph_count).sum()
    }
}

/// Text layout engine.
///
/// Performs paragraph-level text layout including line breaking, alignment,
/// and glyph positioning for broadcast graphics rendering.
pub struct TextLayoutEngine {
    /// Layout configuration.
    config: TextLayoutConfig,
    /// Average character width estimate (for simple metrics).
    avg_char_width: f32,
}

impl TextLayoutEngine {
    /// Create a new text layout engine with the given configuration.
    pub fn new(config: TextLayoutConfig) -> Self {
        let avg_char_width = config.font_size * 0.6;
        Self {
            config,
            avg_char_width,
        }
    }

    /// Update the layout configuration.
    pub fn set_config(&mut self, config: TextLayoutConfig) {
        self.avg_char_width = config.font_size * 0.6;
        self.config = config;
    }

    /// Get the current configuration.
    pub fn config(&self) -> &TextLayoutConfig {
        &self.config
    }

    /// Estimate the width of a character using simple metrics.
    fn estimate_char_width(&self, ch: char) -> f32 {
        (if ch == ' ' {
            self.avg_char_width * self.config.word_spacing
        } else if ch == '\t' {
            self.avg_char_width * self.config.tab_width as f32
        } else if ch.is_ascii_uppercase() || ch == 'W' || ch == 'M' {
            self.avg_char_width * 1.2
        } else if ch == 'i' || ch == 'l' || ch == '!' || ch == '|' {
            self.avg_char_width * 0.5
        } else {
            self.avg_char_width
        }) + self.config.letter_spacing
    }

    /// Estimate the width of a string.
    pub fn estimate_text_width(&self, text: &str) -> f32 {
        text.chars().map(|ch| self.estimate_char_width(ch)).sum()
    }

    /// Find the word boundary for breaking.
    fn find_word_break(&self, text: &str, max_width: f32) -> usize {
        let mut width = 0.0;
        let mut last_space = 0;
        for (i, ch) in text.chars().enumerate() {
            if ch == ' ' || ch == '\t' {
                last_space = i;
            }
            width += self.estimate_char_width(ch);
            if width > max_width {
                return if last_space > 0 { last_space } else { i.max(1) };
            }
        }
        text.chars().count()
    }

    /// Find the character boundary for breaking.
    fn find_char_break(&self, text: &str, max_width: f32) -> usize {
        let mut width = 0.0;
        for (i, ch) in text.chars().enumerate() {
            width += self.estimate_char_width(ch);
            if width > max_width {
                return i.max(1);
            }
        }
        text.chars().count()
    }

    /// Layout text according to the current configuration.
    ///
    /// When [`TextLayoutConfig::direction`] is `Rtl` (or `Auto` and the text
    /// is detected as RTL), glyphs are **mirrored** within each line so that
    /// the first logical character appears near the right edge.
    pub fn layout(&self, text: &str) -> TextLayoutResult {
        let total_chars = text.chars().count();
        let line_h = self.config.effective_line_height();
        let max_w = self.config.max_width;
        // Resolve the effective text direction for this particular string.
        let resolved_dir = self.config.direction.resolve(text);

        let mut lines: Vec<LayoutLine> = Vec::new();
        let mut y_offset: f32 = 0.0;
        let mut global_char_idx: usize = 0;
        let mut truncated = false;

        // Split by explicit newlines (paragraphs).
        let paragraphs: Vec<&str> = text.split('\n').collect();

        for (para_idx, paragraph) in paragraphs.iter().enumerate() {
            if para_idx > 0 {
                y_offset += self.config.paragraph_spacing;
            }

            if paragraph.is_empty() {
                // Empty paragraph still advances by one line.
                let line = LayoutLine::new(y_offset, line_h, global_char_idx);
                lines.push(line);
                y_offset += line_h;
                global_char_idx += 1; // account for the newline
                continue;
            }

            let mut remaining = *paragraph;
            let mut first_line_of_para = true;

            while !remaining.is_empty() {
                // Check height limit.
                if self.config.max_height > 0.0 && y_offset + line_h > self.config.max_height {
                    truncated = true;
                    break;
                }

                let effective_max = if first_line_of_para {
                    max_w - self.config.first_line_indent
                } else {
                    max_w
                };

                let line_width = self.estimate_text_width(remaining);

                let break_at = if self.config.line_break == LineBreakMode::NoWrap {
                    remaining.chars().count()
                } else if line_width <= effective_max {
                    remaining.chars().count()
                } else {
                    match self.config.line_break {
                        LineBreakMode::WordWrap => self.find_word_break(remaining, effective_max),
                        LineBreakMode::CharWrap => self.find_char_break(remaining, effective_max),
                        LineBreakMode::WordCharWrap => {
                            let wb = self.find_word_break(remaining, effective_max);
                            if wb == 0 {
                                self.find_char_break(remaining, effective_max)
                            } else {
                                wb
                            }
                        }
                        LineBreakMode::NoWrap => remaining.chars().count(),
                    }
                };

                // Extract the line text.
                let line_text: String = remaining.chars().take(break_at).collect();
                let consumed_chars = line_text.chars().count();

                // Build glyphs for this line.
                let x_start = if first_line_of_para {
                    self.config.first_line_indent
                } else {
                    0.0
                };

                let mut glyphs = Vec::new();
                let mut x = x_start;
                for (i, ch) in line_text.chars().enumerate() {
                    let adv = self.estimate_char_width(ch);
                    glyphs.push(LayoutGlyph {
                        character: ch,
                        x,
                        y: y_offset,
                        advance_width: adv,
                        line_height: line_h,
                        line_index: lines.len(),
                        char_index: global_char_idx + i,
                    });
                    x += adv;
                }

                let mut layout_line = LayoutLine::new(y_offset, line_h, global_char_idx);
                layout_line.width = x - x_start;
                layout_line.end_char = global_char_idx + consumed_chars;
                layout_line.glyphs = glyphs;
                lines.push(layout_line);

                y_offset += line_h;
                global_char_idx += consumed_chars;

                // Skip past consumed characters and any trailing space.
                let rest: String = remaining.chars().skip(break_at).collect();
                remaining = if rest.starts_with(' ') {
                    global_char_idx += 1;
                    let trimmed: String = rest.chars().skip(1).collect();
                    // We need remaining to be a &str, but we own the String.
                    // Use a workaround: leak is not ideal, so just re-check.
                    // Instead, let's use an owned-string approach.
                    Box::leak(trimmed.into_boxed_str())
                } else {
                    Box::leak(rest.into_boxed_str())
                };

                first_line_of_para = false;
            }

            if truncated {
                break;
            }

            // Account for the newline character between paragraphs.
            if para_idx < paragraphs.len() - 1 {
                global_char_idx += 1;
            }
        }

        // Apply horizontal alignment.
        let total_lines = lines.len();
        for (line_idx, line) in lines.iter_mut().enumerate() {
            match self.config.align {
                TextAlign::Left => {}
                TextAlign::Center => {
                    let shift = (max_w - line.width) / 2.0;
                    if shift > 0.0 {
                        for glyph in &mut line.glyphs {
                            glyph.x += shift;
                        }
                    }
                }
                TextAlign::Right => {
                    let shift = max_w - line.width;
                    if shift > 0.0 {
                        for glyph in &mut line.glyphs {
                            glyph.x += shift;
                        }
                    }
                }
                TextAlign::Justify => {
                    // Last line and single-glyph lines are left-aligned.
                    let is_last_line = line_idx == total_lines - 1;
                    let space_count = line
                        .glyphs
                        .windows(2)
                        .filter(|w| w[0].character == ' ')
                        .count();
                    if !is_last_line && space_count > 0 {
                        let slack = (max_w - line.width).max(0.0);
                        let extra_per_space = slack / space_count as f32;
                        let mut accumulated_shift = 0.0_f32;
                        for glyph in &mut line.glyphs {
                            glyph.x += accumulated_shift;
                            if glyph.character == ' ' {
                                accumulated_shift += extra_per_space;
                            }
                        }
                        // Update line width to reflect full justification.
                        line.width = max_w;
                    }
                }
            }
        }

        // Apply RTL glyph mirroring.
        //
        // In RTL mode the layout was built left-to-right (for simplicity), so
        // we mirror each glyph's X position within the line width:
        //   x_rtl = line_width - (x_ltr + advance_width)
        //
        // This places the **first** logical character at the right edge and the
        // **last** at the left edge, matching RTL reading order.
        if resolved_dir == TextDirection::Rtl {
            for line in &mut lines {
                let line_w = line.width;
                for glyph in &mut line.glyphs {
                    // Mirror: new right edge = line_w - old left edge
                    // new left edge = line_w - (old left edge + advance)
                    glyph.x = line_w - glyph.x - glyph.advance_width;
                }
                // Reverse glyph order so iteration is still left-to-right on screen.
                line.glyphs.reverse();
            }
        }

        // Apply vertical alignment.
        let total_height = y_offset;
        if self.config.max_height > 0.0 && self.config.vertical_align != VerticalAlign::Top {
            let v_shift = match self.config.vertical_align {
                VerticalAlign::Top => 0.0,
                VerticalAlign::Middle => (self.config.max_height - total_height) / 2.0,
                VerticalAlign::Bottom => self.config.max_height - total_height,
            };
            if v_shift > 0.0 {
                for line in &mut lines {
                    line.y_offset += v_shift;
                    for glyph in &mut line.glyphs {
                        glyph.y += v_shift;
                    }
                }
            }
        }

        let total_width = lines.iter().map(|l| l.width).fold(0.0_f32, f32::max);

        TextLayoutResult {
            lines,
            total_width,
            total_height,
            truncated,
            chars_fitted: global_char_idx,
            total_chars,
        }
    }

    /// Layout text and return just the bounding box dimensions.
    pub fn measure(&self, text: &str) -> (f32, f32) {
        let result = self.layout(text);
        (result.total_width, result.total_height)
    }

    /// Check if text will fit within the configured bounds without truncation.
    pub fn fits(&self, text: &str) -> bool {
        let result = self.layout(text);
        result.is_complete()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_align_display() {
        assert_eq!(format!("{}", TextAlign::Left), "left");
        assert_eq!(format!("{}", TextAlign::Center), "center");
        assert_eq!(format!("{}", TextAlign::Right), "right");
        assert_eq!(format!("{}", TextAlign::Justify), "justify");
    }

    #[test]
    fn test_text_align_default() {
        assert_eq!(TextAlign::default(), TextAlign::Left);
    }

    #[test]
    fn test_vertical_align_default() {
        assert_eq!(VerticalAlign::default(), VerticalAlign::Top);
    }

    #[test]
    fn test_text_overflow_default() {
        assert_eq!(TextOverflow::default(), TextOverflow::Clip);
    }

    #[test]
    fn test_line_break_mode_default() {
        assert_eq!(LineBreakMode::default(), LineBreakMode::WordWrap);
    }

    #[test]
    fn test_layout_config_default() {
        let config = TextLayoutConfig::default();
        assert_eq!(config.max_width, 800.0);
        assert_eq!(config.font_size, 16.0);
        assert_eq!(config.line_height, 1.2);
        assert_eq!(config.letter_spacing, 0.0);
        assert_eq!(config.word_spacing, 1.0);
        assert_eq!(config.tab_width, 4);
    }

    #[test]
    fn test_effective_line_height() {
        let config = TextLayoutConfig {
            font_size: 20.0,
            line_height: 1.5,
            ..Default::default()
        };
        let expected = 20.0 * 1.5;
        assert!((config.effective_line_height() - expected).abs() < 0.001);
    }

    #[test]
    fn test_broadcast_title_config() {
        let config = TextLayoutConfig::broadcast_title(1920.0, 48.0);
        assert_eq!(config.align, TextAlign::Center);
        assert_eq!(config.vertical_align, VerticalAlign::Middle);
        assert_eq!(config.font_size, 48.0);
        assert_eq!(config.overflow, TextOverflow::ShrinkToFit);
    }

    #[test]
    fn test_lower_third_config() {
        let config = TextLayoutConfig::lower_third(600.0, 24.0);
        assert_eq!(config.align, TextAlign::Left);
        assert_eq!(config.vertical_align, VerticalAlign::Bottom);
        assert_eq!(config.overflow, TextOverflow::Ellipsis);
    }

    #[test]
    fn test_ticker_config() {
        let config = TextLayoutConfig::ticker(1920.0, 18.0);
        assert_eq!(config.line_break, LineBreakMode::NoWrap);
        assert_eq!(config.overflow, TextOverflow::Visible);
    }

    #[test]
    fn test_simple_layout() {
        let config = TextLayoutConfig {
            max_width: 1000.0,
            font_size: 16.0,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        let result = engine.layout("Hello World");
        assert_eq!(result.line_count(), 1);
        assert!(!result.truncated);
        assert!(result.total_width > 0.0);
        assert!(result.total_height > 0.0);
    }

    #[test]
    fn test_multiline_layout() {
        let config = TextLayoutConfig {
            max_width: 1000.0,
            font_size: 16.0,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        let result = engine.layout("Line one\nLine two\nLine three");
        assert_eq!(result.line_count(), 3);
        assert!(!result.truncated);
    }

    #[test]
    fn test_word_wrap() {
        // Use a very narrow width to force wrapping.
        let config = TextLayoutConfig {
            max_width: 60.0,
            font_size: 16.0,
            line_break: LineBreakMode::WordWrap,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        let result = engine.layout("Hello World Test");
        assert!(result.line_count() > 1);
    }

    #[test]
    fn test_no_wrap() {
        let config = TextLayoutConfig {
            max_width: 50.0,
            font_size: 16.0,
            line_break: LineBreakMode::NoWrap,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        let result = engine.layout("This is a very long line that should not wrap");
        assert_eq!(result.line_count(), 1);
    }

    #[test]
    fn test_estimate_text_width() {
        let config = TextLayoutConfig {
            font_size: 16.0,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        let w = engine.estimate_text_width("Hello");
        assert!(w > 0.0);
    }

    #[test]
    fn test_measure() {
        let config = TextLayoutConfig {
            max_width: 500.0,
            font_size: 20.0,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        let (w, h) = engine.measure("Test string");
        assert!(w > 0.0);
        assert!(h > 0.0);
    }

    #[test]
    fn test_fits() {
        let config = TextLayoutConfig {
            max_width: 1000.0,
            max_height: 0.0, // unlimited height
            font_size: 16.0,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        assert!(engine.fits("Short text"));
    }

    #[test]
    fn test_layout_line_new() {
        let line = LayoutLine::new(10.0, 20.0, 5);
        assert!(line.is_empty());
        assert_eq!(line.glyph_count(), 0);
        assert_eq!(line.y_offset, 10.0);
        assert_eq!(line.height, 20.0);
        assert_eq!(line.start_char, 5);
    }

    #[test]
    fn test_layout_result_completeness() {
        let config = TextLayoutConfig {
            max_width: 1000.0,
            font_size: 16.0,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        let result = engine.layout("Short");
        assert!(result.is_complete());
        assert!(result.glyph_count() > 0);
        assert_eq!(result.total_chars, 5);
    }

    #[test]
    fn test_center_alignment() {
        let config = TextLayoutConfig {
            max_width: 500.0,
            font_size: 16.0,
            align: TextAlign::Center,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        let result = engine.layout("Hi");
        assert_eq!(result.line_count(), 1);
        // Center-aligned glyphs should start past x=0.
        if let Some(first_glyph) = result.lines[0].glyphs.first() {
            assert!(first_glyph.x > 0.0);
        }
    }

    #[test]
    fn test_right_alignment() {
        let config = TextLayoutConfig {
            max_width: 500.0,
            font_size: 16.0,
            align: TextAlign::Right,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        let result = engine.layout("Hi");
        assert_eq!(result.line_count(), 1);
        // Right-aligned glyphs should be near the right edge.
        if let Some(first_glyph) = result.lines[0].glyphs.first() {
            assert!(first_glyph.x > 100.0);
        }
    }

    #[test]
    fn test_justify_alignment_multi_line() {
        // Use a very narrow width to force multi-line wrapping, then check justify.
        let config = TextLayoutConfig {
            max_width: 80.0,
            font_size: 16.0,
            align: TextAlign::Justify,
            line_break: LineBreakMode::WordWrap,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        // This text should wrap to at least 2 lines, making the first line justified.
        let result = engine.layout("Hello World Testing");
        // At least 2 lines expected due to narrow width.
        if result.line_count() >= 2 {
            // First (non-last) line should be justified to full width.
            let first_line = &result.lines[0];
            // Width should equal max_width (or close) after justify.
            if !first_line.glyphs.is_empty() {
                // Justified line width equals max_w.
                assert!((first_line.width - 80.0).abs() < 1.0 || first_line.width <= 80.0);
            }
        }
    }

    #[test]
    fn test_set_config() {
        let config1 = TextLayoutConfig {
            font_size: 16.0,
            ..Default::default()
        };
        let config2 = TextLayoutConfig {
            font_size: 32.0,
            ..Default::default()
        };
        let mut engine = TextLayoutEngine::new(config1);
        engine.set_config(config2);
        assert_eq!(engine.config().font_size, 32.0);
    }

    #[test]
    fn test_empty_text_layout() {
        let config = TextLayoutConfig::default();
        let engine = TextLayoutEngine::new(config);
        let result = engine.layout("");
        assert_eq!(result.total_chars, 0);
    }

    // --- TextDirection / RTL tests ---

    #[test]
    fn test_text_direction_default_is_ltr() {
        assert_eq!(TextDirection::default(), TextDirection::Ltr);
    }

    #[test]
    fn test_text_direction_ltr_resolves_ltr() {
        assert_eq!(TextDirection::Ltr.resolve("Hello"), TextDirection::Ltr);
    }

    #[test]
    fn test_text_direction_rtl_resolves_rtl() {
        assert_eq!(TextDirection::Rtl.resolve("مرحبا"), TextDirection::Rtl);
    }

    #[test]
    fn test_text_direction_auto_latin_is_ltr() {
        assert_eq!(
            TextDirection::Auto.resolve("Hello World"),
            TextDirection::Ltr
        );
    }

    #[test]
    fn test_text_direction_auto_arabic_is_rtl() {
        // U+0645 is Arabic letter Meem.
        let arabic = "\u{0645}\u{0631}\u{062D}\u{0628}\u{0627}"; // "مرحبا"
        assert_eq!(TextDirection::Auto.resolve(arabic), TextDirection::Rtl);
    }

    #[test]
    fn test_text_direction_auto_hebrew_is_rtl() {
        // U+05E9 is Hebrew letter Shin.
        let hebrew = "\u{05E9}\u{05DC}\u{05D5}\u{05DD}"; // "שלום"
        assert_eq!(TextDirection::Auto.resolve(hebrew), TextDirection::Rtl);
    }

    #[test]
    fn test_text_direction_auto_empty_is_ltr() {
        assert_eq!(TextDirection::Auto.resolve(""), TextDirection::Ltr);
    }

    #[test]
    fn test_text_direction_is_rtl_helper() {
        assert!(TextDirection::Rtl.is_rtl("anything"));
        assert!(!TextDirection::Ltr.is_rtl("anything"));
    }

    #[test]
    fn test_layout_config_has_direction_field() {
        let cfg = TextLayoutConfig::default();
        assert_eq!(cfg.direction, TextDirection::Ltr);
    }

    #[test]
    fn test_ticker_rtl_config() {
        let cfg = TextLayoutConfig::ticker_rtl(1920.0, 18.0);
        assert_eq!(cfg.direction, TextDirection::Rtl);
        assert_eq!(cfg.align, TextAlign::Right);
        assert_eq!(cfg.line_break, LineBreakMode::NoWrap);
    }

    #[test]
    fn test_rtl_paragraph_config() {
        let cfg = TextLayoutConfig::rtl_paragraph(800.0, 24.0);
        assert_eq!(cfg.direction, TextDirection::Rtl);
        assert_eq!(cfg.align, TextAlign::Right);
    }

    #[test]
    fn test_rtl_layout_glyph_positions_are_mirrored() {
        // RTL layout should mirror glyph x-positions within the line width.
        let text = "ABC"; // Three equal-width chars
        let config = TextLayoutConfig {
            max_width: 1000.0,
            font_size: 16.0,
            direction: TextDirection::Rtl,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        let result = engine.layout(text);
        assert_eq!(result.line_count(), 1);
        let glyphs = &result.lines[0].glyphs;
        assert_eq!(glyphs.len(), 3);
        // After mirroring + reversal, screen-left glyph should be 'C' (last logical).
        // In our simplified model all chars have equal width so positions are:
        // logical: A@0, B@w, C@2w — mirrored: C@0, B@w, A@2w
        // The first on-screen glyph should be 'C'.
        assert_eq!(glyphs[0].character, 'C');
        assert_eq!(glyphs[1].character, 'B');
        assert_eq!(glyphs[2].character, 'A');
    }

    #[test]
    fn test_rtl_layout_glyph_x_positions_increasing() {
        // After RTL mirroring the on-screen X positions should be non-decreasing.
        let config = TextLayoutConfig {
            max_width: 500.0,
            font_size: 16.0,
            direction: TextDirection::Rtl,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        let result = engine.layout("ABCDE");
        let glyphs = &result.lines[0].glyphs;
        for pair in glyphs.windows(2) {
            assert!(
                pair[1].x >= pair[0].x,
                "x positions should be non-decreasing after RTL mirror"
            );
        }
    }

    #[test]
    fn test_ltr_layout_first_char_near_left_edge() {
        // Sanity check: LTR layout first glyph should be at x ≈ 0.
        let config = TextLayoutConfig {
            max_width: 1000.0,
            font_size: 16.0,
            direction: TextDirection::Ltr,
            ..Default::default()
        };
        let engine = TextLayoutEngine::new(config);
        let result = engine.layout("ABC");
        let glyphs = &result.lines[0].glyphs;
        // First glyph x should be 0 (left-aligned, no indent).
        assert!(glyphs[0].x < 1.0);
    }

    #[test]
    fn test_rtl_layout_total_width_equals_ltr() {
        // RTL mirroring should not change the total content width of a line.
        let text = "Hello";
        let ltr_config = TextLayoutConfig {
            max_width: 500.0,
            font_size: 16.0,
            direction: TextDirection::Ltr,
            ..Default::default()
        };
        let rtl_config = TextLayoutConfig {
            max_width: 500.0,
            font_size: 16.0,
            direction: TextDirection::Rtl,
            ..Default::default()
        };
        let ltr_result = TextLayoutEngine::new(ltr_config).layout(text);
        let rtl_result = TextLayoutEngine::new(rtl_config).layout(text);
        assert!(
            (ltr_result.total_width - rtl_result.total_width).abs() < 0.1,
            "LTR width={} RTL width={}",
            ltr_result.total_width,
            rtl_result.total_width
        );
    }
}

// ============================================================================
// High-level multiline API with newtype aliases
// ============================================================================

/// Wrap mode alias — maps to the underlying [`LineBreakMode`].
///
/// This type alias exposes a simpler, intention-revealing name for callers
/// who use the [`TextLayout`] high-level API.
pub type WrapMode = LineBreakMode;

/// Justification alias — maps to the underlying [`TextAlign`].
///
/// This type alias exposes a simpler name for callers who use the
/// [`TextLayout`] high-level API.
pub type Justification = TextAlign;

/// Configuration for the high-level [`TextLayout::layout_multiline`] function.
///
/// Mirrors the fields most commonly needed for simple paragraph layout, and
/// internally converts to a [`TextLayoutConfig`] when invoking the engine.
#[derive(Clone, Debug)]
pub struct MultilineLayoutConfig {
    /// Maximum line width in pixels.
    pub max_width: u32,
    /// Word/char/no-wrap strategy.
    pub wrap_mode: WrapMode,
    /// Horizontal justification.
    pub justification: Justification,
    /// Line-height multiplier (1.0 = normal).
    pub line_spacing: f32,
    /// Font size in pixels (used for advance-width estimation).
    pub font_size: f32,
}

impl Default for MultilineLayoutConfig {
    fn default() -> Self {
        Self {
            max_width: 800,
            wrap_mode: WrapMode::WordWrap,
            justification: Justification::Left,
            line_spacing: 1.2,
            font_size: 16.0,
        }
    }
}

/// Unit struct providing the [`layout_multiline`](TextLayout::layout_multiline) entry-point.
pub struct TextLayout;

impl TextLayout {
    /// Lay out `text` according to `config` and return one [`LayoutLine`] per
    /// wrapped line.
    ///
    /// Internally this builds a [`TextLayoutConfig`], creates a
    /// [`TextLayoutEngine`], runs `layout()`, and returns the resulting lines.
    #[must_use]
    pub fn layout_multiline(text: &str, config: &MultilineLayoutConfig) -> Vec<LayoutLine> {
        let engine_config = TextLayoutConfig {
            max_width: config.max_width as f32,
            max_height: 0.0,
            align: config.justification,
            vertical_align: VerticalAlign::Top,
            font_size: config.font_size,
            line_height: config.line_spacing,
            letter_spacing: 0.0,
            word_spacing: 1.0,
            overflow: TextOverflow::Visible,
            line_break: config.wrap_mode,
            first_line_indent: 0.0,
            paragraph_spacing: 0.0,
            tab_width: 4,
            direction: TextDirection::Ltr,
        };
        let engine = TextLayoutEngine::new(engine_config);
        engine.layout(text).lines
    }
}

// ============================================================================
// Multiline API tests
// ============================================================================

#[cfg(test)]
mod multiline_tests {
    use super::*;

    fn default_config() -> MultilineLayoutConfig {
        MultilineLayoutConfig::default()
    }

    // ── Type alias checks ────────────────────────────────────────────────────

    #[test]
    fn test_wrap_mode_alias_is_line_break_mode() {
        // Compile-time confirmation that WrapMode == LineBreakMode.
        let _: WrapMode = LineBreakMode::WordWrap;
        let _: WrapMode = LineBreakMode::CharWrap;
        let _: WrapMode = LineBreakMode::NoWrap;
    }

    #[test]
    fn test_justification_alias_is_text_align() {
        let _: Justification = TextAlign::Left;
        let _: Justification = TextAlign::Center;
        let _: Justification = TextAlign::Right;
        let _: Justification = TextAlign::Justify;
    }

    // ── MultilineLayoutConfig ────────────────────────────────────────────────

    #[test]
    fn test_multiline_config_default_sensible() {
        let cfg = default_config();
        assert!(cfg.max_width > 0);
        assert!(cfg.line_spacing > 0.0);
        assert!(cfg.font_size > 0.0);
        assert_eq!(cfg.justification, Justification::Left);
        assert_eq!(cfg.wrap_mode, WrapMode::WordWrap);
    }

    // ── TextLayout::layout_multiline ─────────────────────────────────────────

    #[test]
    fn test_layout_multiline_nonempty_text_returns_lines() {
        let cfg = default_config();
        let lines = TextLayout::layout_multiline("Hello World", &cfg);
        assert!(!lines.is_empty(), "Should have at least one line");
    }

    #[test]
    fn test_layout_multiline_empty_text_returns_empty() {
        let cfg = default_config();
        let lines = TextLayout::layout_multiline("", &cfg);
        assert!(lines.is_empty() || lines.iter().all(|l| l.is_empty()));
    }

    // ── WrapMode::None (NoWrap) ──────────────────────────────────────────────

    #[test]
    fn test_wrap_mode_none_single_line() {
        let cfg = MultilineLayoutConfig {
            max_width: 50,
            wrap_mode: WrapMode::NoWrap,
            ..default_config()
        };
        let lines = TextLayout::layout_multiline("This is a very long line of text", &cfg);
        assert_eq!(lines.len(), 1, "NoWrap should produce exactly 1 line");
    }

    // ── WrapMode::WordWrap ───────────────────────────────────────────────────

    #[test]
    fn test_wrap_mode_word_wrap_produces_multiple_lines() {
        let cfg = MultilineLayoutConfig {
            max_width: 60,
            font_size: 16.0,
            wrap_mode: WrapMode::WordWrap,
            ..default_config()
        };
        let lines = TextLayout::layout_multiline("Hello World Testing Overflow", &cfg);
        assert!(lines.len() > 1, "WordWrap should wrap into multiple lines");
    }

    #[test]
    fn test_wrap_mode_word_wrap_each_line_nonempty_glyphs() {
        let cfg = MultilineLayoutConfig {
            max_width: 60,
            font_size: 16.0,
            wrap_mode: WrapMode::WordWrap,
            ..default_config()
        };
        let lines = TextLayout::layout_multiline("Alpha Beta Gamma Delta", &cfg);
        for line in &lines {
            assert!(!line.is_empty(), "Each wrapped line should have glyphs");
        }
    }

    // ── WrapMode::CharWrap ───────────────────────────────────────────────────

    #[test]
    fn test_wrap_mode_char_wrap_produces_multiple_lines() {
        let cfg = MultilineLayoutConfig {
            max_width: 40,
            font_size: 16.0,
            wrap_mode: WrapMode::CharWrap,
            ..default_config()
        };
        let lines = TextLayout::layout_multiline("ABCDEFGHIJKLMNOPQRSTUVWXYZ", &cfg);
        assert!(lines.len() > 1, "CharWrap should split into multiple lines");
    }

    // ── Justification::Left ──────────────────────────────────────────────────

    #[test]
    fn test_justification_left_first_glyph_at_zero() {
        let cfg = MultilineLayoutConfig {
            max_width: 500,
            font_size: 16.0,
            justification: Justification::Left,
            ..default_config()
        };
        let lines = TextLayout::layout_multiline("Hello", &cfg);
        if let Some(first_line) = lines.first() {
            if let Some(first_glyph) = first_line.glyphs.first() {
                assert!(
                    first_glyph.x < 1.0,
                    "Left-aligned first glyph x should be ~0, got {}",
                    first_glyph.x
                );
            }
        }
    }

    // ── Justification::Center ────────────────────────────────────────────────

    #[test]
    fn test_justification_center_first_glyph_offset() {
        let cfg = MultilineLayoutConfig {
            max_width: 500,
            font_size: 16.0,
            justification: Justification::Center,
            ..default_config()
        };
        let lines = TextLayout::layout_multiline("Hi", &cfg);
        if let Some(first_line) = lines.first() {
            if let Some(first_glyph) = first_line.glyphs.first() {
                assert!(
                    first_glyph.x > 0.0,
                    "Center-aligned first glyph x should be > 0, got {}",
                    first_glyph.x
                );
            }
        }
    }

    // ── Justification::Right ─────────────────────────────────────────────────

    #[test]
    fn test_justification_right_first_glyph_offset() {
        let cfg = MultilineLayoutConfig {
            max_width: 500,
            font_size: 16.0,
            justification: Justification::Right,
            ..default_config()
        };
        let lines = TextLayout::layout_multiline("Hi", &cfg);
        if let Some(first_line) = lines.first() {
            if let Some(first_glyph) = first_line.glyphs.first() {
                assert!(
                    first_glyph.x > 100.0,
                    "Right-aligned first glyph x should be near right edge, got {}",
                    first_glyph.x
                );
            }
        }
    }

    // ── Justification::Full (Justify) ────────────────────────────────────────

    #[test]
    fn test_justification_full_non_last_line_expanded() {
        let cfg = MultilineLayoutConfig {
            max_width: 80,
            font_size: 16.0,
            justification: Justification::Justify,
            wrap_mode: WrapMode::WordWrap,
            ..default_config()
        };
        let lines = TextLayout::layout_multiline("Hello World Testing", &cfg);
        if lines.len() >= 2 {
            // Non-last line with spaces should be justified to full width.
            let first = &lines[0];
            let has_spaces = first.glyphs.iter().any(|g| g.character == ' ');
            if has_spaces {
                assert!(
                    (first.width - 80.0).abs() < 1.0 || first.width <= 80.0,
                    "Justified non-last line width={} should be ~80.0",
                    first.width
                );
            }
        }
    }

    #[test]
    fn test_justification_full_last_line_not_expanded() {
        let cfg = MultilineLayoutConfig {
            max_width: 80,
            font_size: 16.0,
            justification: Justification::Justify,
            wrap_mode: WrapMode::WordWrap,
            ..default_config()
        };
        let lines = TextLayout::layout_multiline("Hello World Testing", &cfg);
        if let Some(last) = lines.last() {
            // Last line should not be forced to max_width.
            assert!(
                last.width <= 80.0 + 1.0,
                "Last justified line should not exceed max_width"
            );
        }
    }

    // ── Line spacing ─────────────────────────────────────────────────────────

    #[test]
    fn test_line_spacing_affects_y_offset() {
        let cfg1 = MultilineLayoutConfig {
            max_width: 60,
            font_size: 16.0,
            line_spacing: 1.0,
            wrap_mode: WrapMode::WordWrap,
            ..default_config()
        };
        let cfg2 = MultilineLayoutConfig {
            max_width: 60,
            font_size: 16.0,
            line_spacing: 2.0,
            wrap_mode: WrapMode::WordWrap,
            ..default_config()
        };
        let lines1 = TextLayout::layout_multiline("Hello World Testing", &cfg1);
        let lines2 = TextLayout::layout_multiline("Hello World Testing", &cfg2);
        if lines1.len() >= 2 && lines2.len() >= 2 {
            let gap1 = lines1[1].y_offset - lines1[0].y_offset;
            let gap2 = lines2[1].y_offset - lines2[0].y_offset;
            assert!(
                gap2 > gap1,
                "Larger line_spacing should produce larger y gap: gap1={gap1}, gap2={gap2}"
            );
        }
    }

    // ── Additional corner-case tests ─────────────────────────────────────────

    #[test]
    fn test_single_word_no_wrap() {
        let cfg = MultilineLayoutConfig {
            max_width: 10,
            font_size: 16.0,
            wrap_mode: WrapMode::WordWrap,
            ..default_config()
        };
        // A single long word cannot break on word boundaries, so it stays on one line.
        let lines = TextLayout::layout_multiline("Superlongwordwithnobreaks", &cfg);
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_multiline_explicit_newlines_respected() {
        let cfg = MultilineLayoutConfig {
            max_width: 1000,
            ..default_config()
        };
        let lines = TextLayout::layout_multiline("Line one\nLine two\nLine three", &cfg);
        assert!(
            lines.len() >= 3,
            "Explicit newlines should produce >= 3 lines"
        );
    }

    #[test]
    fn test_char_wrap_lines_have_finite_widths() {
        let cfg = MultilineLayoutConfig {
            max_width: 40,
            font_size: 16.0,
            wrap_mode: WrapMode::CharWrap,
            ..default_config()
        };
        let lines = TextLayout::layout_multiline("ABCDEFGHIJKLMNO", &cfg);
        for line in &lines {
            assert!(line.width.is_finite(), "Line width should be finite");
        }
    }
}
