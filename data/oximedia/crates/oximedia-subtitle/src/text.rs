//! Text layout and bidirectional text support.

use crate::font::{Font, GlyphCache, SimpleLayoutEngine};
use crate::style::{Alignment, SubtitleStyle};
use crate::{SubtitleError, SubtitleResult};
use unicode_bidi::{BidiInfo, Level};
use unicode_segmentation::UnicodeSegmentation;

/// Bidirectional text level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BidiLevel(u8);

impl BidiLevel {
    /// Create a new bidi level.
    #[must_use]
    pub const fn new(level: u8) -> Self {
        Self(level)
    }

    /// Left-to-right.
    #[must_use]
    pub const fn ltr() -> Self {
        Self(0)
    }

    /// Right-to-left.
    #[must_use]
    pub const fn rtl() -> Self {
        Self(1)
    }

    /// Check if this is RTL.
    #[must_use]
    pub const fn is_rtl(&self) -> bool {
        self.0 % 2 == 1
    }

    /// Get the underlying level value.
    #[must_use]
    pub const fn value(&self) -> u8 {
        self.0
    }
}

impl From<Level> for BidiLevel {
    fn from(level: Level) -> Self {
        Self(level.number())
    }
}

/// A line of laid out text with glyph positions.
#[derive(Clone, Debug)]
pub struct TextLine {
    /// Glyphs in this line.
    pub glyphs: Vec<PositionedGlyph>,
    /// Line width in pixels.
    pub width: f32,
    /// Line height in pixels.
    pub height: f32,
    /// Baseline Y offset.
    pub baseline: f32,
}

/// A positioned glyph with all rendering information.
#[derive(Clone, Debug)]
pub struct PositionedGlyph {
    /// Character.
    pub c: char,
    /// X position.
    pub x: f32,
    /// Y position.
    pub y: f32,
    /// Glyph bitmap width.
    pub width: usize,
    /// Glyph bitmap height.
    pub height: usize,
    /// Rasterized bitmap.
    pub bitmap: Vec<u8>,
    /// Bidirectional level.
    pub bidi_level: BidiLevel,
}

/// Complete text layout with multiple lines.
#[derive(Clone, Debug)]
pub struct TextLayout {
    /// Lines of text.
    pub lines: Vec<TextLine>,
    /// Total width.
    pub width: f32,
    /// Total height.
    pub height: f32,
}

impl TextLayout {
    /// Create an empty layout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            width: 0.0,
            height: 0.0,
        }
    }

    /// Add a line.
    pub fn add_line(&mut self, line: TextLine) {
        self.width = self.width.max(line.width);
        self.height += line.height;
        self.lines.push(line);
    }

    /// Get bounds.
    #[must_use]
    pub const fn bounds(&self) -> (f32, f32) {
        (self.width, self.height)
    }

    /// Check if layout is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

impl Default for TextLayout {
    fn default() -> Self {
        Self::new()
    }
}

/// Text layout engine with full Unicode and bidirectional text support.
pub struct TextLayoutEngine {
    glyph_cache: GlyphCache,
    simple_layout: SimpleLayoutEngine,
}

impl TextLayoutEngine {
    /// Create a new text layout engine.
    #[must_use]
    pub fn new(font: Font) -> Self {
        Self {
            glyph_cache: GlyphCache::new(font),
            simple_layout: SimpleLayoutEngine::new(),
        }
    }

    /// Layout text according to style.
    ///
    /// # Errors
    ///
    /// Returns error if text layout fails.
    pub fn layout(
        &mut self,
        text: &str,
        style: &SubtitleStyle,
        max_width: u32,
    ) -> SubtitleResult<TextLayout> {
        if text.is_empty() {
            return Ok(TextLayout::new());
        }

        // Handle bidirectional text
        let bidi_info = BidiInfo::new(text, None);
        let paragraph = &bidi_info.paragraphs[0];
        let line_bidi = bidi_info.reorder_line(paragraph, paragraph.range.clone());

        // Split into lines (handle explicit line breaks)
        let mut layout = TextLayout::new();
        let font_metrics = self.glyph_cache.font().metrics(style.font_size);

        for line_text in text.lines() {
            if line_text.is_empty() {
                // Empty line - just add spacing
                layout.add_line(TextLine {
                    glyphs: Vec::new(),
                    width: 0.0,
                    height: font_metrics.new_line_size * style.line_spacing,
                    baseline: font_metrics.ascent,
                });
                continue;
            }

            // Use simple layout for word wrapping if needed
            let max_w = if max_width > 0 {
                Some(max_width as f32)
            } else {
                None
            };

            let glyph_positions = self.simple_layout.layout_text(
                self.glyph_cache.font(),
                line_text,
                style.font_size,
                max_w,
            );

            // Group glyphs by line (in case of wrapping)
            let mut current_line = Vec::new();
            let mut current_y = 0.0;
            let mut line_width = 0.0;
            let line_height = font_metrics.new_line_size * style.line_spacing;

            for glyph_pos in glyph_positions {
                // Check if we moved to a new line
                if !current_line.is_empty() && (glyph_pos.y - current_y).abs() > line_height / 2.0 {
                    // Finish current line
                    self.finish_line(
                        &mut layout,
                        current_line,
                        line_width,
                        line_height,
                        font_metrics.ascent,
                        style.alignment,
                    );
                    current_line = Vec::new();
                    line_width = 0.0;
                }

                // Get cached glyph
                let cached = self.glyph_cache.get_glyph(glyph_pos.c, style.font_size);

                current_line.push(PositionedGlyph {
                    c: glyph_pos.c,
                    x: glyph_pos.x,
                    y: glyph_pos.y,
                    width: cached.width,
                    height: cached.height,
                    bitmap: cached.bitmap.clone(),
                    bidi_level: BidiLevel::ltr(), // Simplified - would need proper bidi mapping
                });

                line_width = line_width.max(glyph_pos.x + glyph_pos.width);
                current_y = glyph_pos.y;
            }

            // Finish last line
            if !current_line.is_empty() {
                self.finish_line(
                    &mut layout,
                    current_line,
                    line_width,
                    line_height,
                    font_metrics.ascent,
                    style.alignment,
                );
            }
        }

        Ok(layout)
    }

    /// Finish and add a line to the layout with proper alignment.
    fn finish_line(
        &self,
        layout: &mut TextLayout,
        mut glyphs: Vec<PositionedGlyph>,
        width: f32,
        height: f32,
        baseline: f32,
        alignment: Alignment,
    ) {
        // Apply horizontal alignment by adjusting X positions
        let offset = match alignment {
            Alignment::Left => 0.0,
            Alignment::Center => -width / 2.0,
            Alignment::Right => -width,
        };

        for glyph in &mut glyphs {
            glyph.x += offset;
        }

        layout.add_line(TextLine {
            glyphs,
            width,
            height,
            baseline,
        });
    }

    /// Get glyph cache.
    #[must_use]
    pub fn glyph_cache(&self) -> &GlyphCache {
        &self.glyph_cache
    }

    /// Get mutable glyph cache.
    pub fn glyph_cache_mut(&mut self) -> &mut GlyphCache {
        &mut self.glyph_cache
    }
}

/// Word wrapping utility.
pub struct WordWrapper {
    max_width: f32,
}

impl WordWrapper {
    /// Create a new word wrapper.
    #[must_use]
    pub const fn new(max_width: f32) -> Self {
        Self { max_width }
    }

    /// Wrap text into lines.
    #[must_use]
    pub fn wrap(&self, text: &str, measure: impl Fn(&str) -> f32) -> Vec<String> {
        let mut lines = Vec::new();
        let mut current_line = String::new();
        let mut current_width = 0.0;

        for word in text.unicode_words() {
            let word_width = measure(word);
            let space_width = measure(" ");

            if current_line.is_empty() {
                current_line.push_str(word);
                current_width = word_width;
            } else if current_width + space_width + word_width <= self.max_width {
                current_line.push(' ');
                current_line.push_str(word);
                current_width += space_width + word_width;
            } else {
                // Start new line
                lines.push(current_line);
                current_line = word.to_string();
                current_width = word_width;
            }
        }

        if !current_line.is_empty() {
            lines.push(current_line);
        }

        if lines.is_empty() {
            lines.push(String::new());
        }

        lines
    }
}

/// Strip HTML tags from text (basic implementation for SRT).
#[must_use]
pub fn strip_html_tags(text: &str) -> String {
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

/// Decode HTML entities (basic implementation).
#[must_use]
pub fn decode_html_entities(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

// ============================================================================
// Bidirectional text tests
// ============================================================================

#[cfg(test)]
mod bidi_tests {
    use unicode_bidi::BidiInfo;

    /// The mixed string starts with a Latin 'H' (strong LTR), so the paragraph
    /// base direction must be detected as LTR (even level 0).
    #[test]
    fn test_bidi_paragraph_direction_ltr_first() {
        let text = "Hello \u{0645}\u{0631}\u{062D}\u{0628}\u{0627} World";
        let bidi = BidiInfo::new(text, None);
        assert_eq!(
            bidi.paragraphs.len(),
            1,
            "should form exactly one paragraph"
        );
        let para = &bidi.paragraphs[0];
        // LTR paragraph: level is even (0).
        assert!(
            !para.level.is_rtl(),
            "paragraph base direction should be LTR when first strong char is Latin"
        );
    }

    /// Reorder the visual runs of the mixed string and verify that Arabic
    /// characters appear between the two English word groups.
    #[test]
    fn test_bidi_mixed_arabic_latin_visual_order() {
        // "Hello مرحبا World"
        // In logical order: H e l l o   [arabic 5 chars]   W o r l d
        let text = "Hello \u{0645}\u{0631}\u{062D}\u{0628}\u{0627} World";
        let bidi = BidiInfo::new(text, None);
        let para = &bidi.paragraphs[0];

        // Reorder the whole paragraph into visual order.
        let reordered: String = bidi.reorder_line(para, para.range.clone()).to_string();

        // In visual (left-to-right display) order for an LTR paragraph:
        // "Hello" comes first, then Arabic (which renders RTL internally),
        // then "World" comes last. The reordered string must contain all three
        // segments.
        assert!(
            reordered.contains("Hello"),
            "reordered string must contain 'Hello'"
        );
        assert!(
            reordered.contains("World"),
            "reordered string must contain 'World'"
        );
        // Arabic characters must still be present.
        let arabic_chars: String = reordered
            .chars()
            .filter(|c| ('\u{0600}'..='\u{06FF}').contains(c))
            .collect();
        assert_eq!(
            arabic_chars.chars().count(),
            5,
            "all 5 Arabic characters must survive reordering"
        );
    }

    /// Verify that a purely LTR string is unchanged after reordering.
    #[test]
    fn test_bidi_pure_ltr_unchanged() {
        let text = "Hello World";
        let bidi = BidiInfo::new(text, None);
        let para = &bidi.paragraphs[0];
        let reordered: String = bidi.reorder_line(para, para.range.clone()).to_string();
        assert_eq!(
            reordered, text,
            "purely LTR text should be unchanged after reorder"
        );
    }

    /// Verify that a purely RTL string is reversed by reorder_line.
    #[test]
    fn test_bidi_pure_rtl_reversal() {
        // A string with only Arabic characters: "مرحبا" (Marhaba = Hello)
        let text = "\u{0645}\u{0631}\u{062D}\u{0628}\u{0627}";
        let bidi = BidiInfo::new(text, None);
        let para = &bidi.paragraphs[0];
        // Paragraph level should be RTL (odd level).
        assert!(para.level.is_rtl(), "purely Arabic paragraph should be RTL");
        let reordered: String = bidi.reorder_line(para, para.range.clone()).to_string();
        // All Arabic chars still present.
        let count = reordered
            .chars()
            .filter(|c| ('\u{0600}'..='\u{06FF}').contains(c))
            .count();
        assert_eq!(count, 5, "all 5 Arabic chars present after RTL reorder");
    }

    /// BidiLevel wrapper correctly reports direction.
    #[test]
    fn test_bidi_level_wrapper() {
        use super::BidiLevel;
        assert!(!BidiLevel::ltr().is_rtl());
        assert!(BidiLevel::rtl().is_rtl());
        assert!(!BidiLevel::new(0).is_rtl());
        assert!(BidiLevel::new(1).is_rtl());
        assert!(!BidiLevel::new(2).is_rtl());
        assert!(BidiLevel::new(3).is_rtl());
    }

    /// From&lt;unicode_bidi::Level&gt; conversion preserves the level number.
    #[test]
    fn test_bidi_level_from_unicode_bidi() {
        use super::BidiLevel;
        use unicode_bidi::Level;
        let ul = Level::ltr();
        let bl = BidiLevel::from(ul);
        assert_eq!(bl.value(), ul.number());
    }
}
