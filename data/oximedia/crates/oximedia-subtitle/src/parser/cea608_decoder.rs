//! CEA-608 closed caption decoder.
//!
//! This module decodes CEA-608 closed caption data from line 21 field data
//! or user data packets into text subtitles.

use crate::{Subtitle, SubtitleError, SubtitleResult};
use std::collections::HashMap;

/// CEA-608 decoder state.
pub struct Cea608Decoder {
    /// Current display mode.
    mode: DisplayMode,
    /// Display buffer (visible text).
    display_buffer: TextBuffer,
    /// Non-display buffer (for pop-on captions).
    non_display_buffer: TextBuffer,
    /// Roll-up buffer.
    rollup_buffer: TextBuffer,
    /// Number of roll-up rows (2, 3, or 4).
    rollup_rows: usize,
    /// Current cursor position.
    cursor_row: usize,
    cursor_col: usize,
    /// Current style.
    current_color: CaptionColor,
    current_italic: bool,
    current_underline: bool,
    /// Accumulated subtitles.
    subtitles: Vec<Subtitle>,
    /// Current subtitle start time.
    current_start_time: Option<i64>,
    /// Last timestamp processed.
    last_timestamp: i64,
}

/// Display mode for CEA-608.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DisplayMode {
    /// Pop-on captions (displayed all at once).
    PopOn,
    /// Roll-up captions (scrolling text).
    RollUp,
    /// Paint-on captions (character by character).
    PaintOn,
}

/// Caption color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum CaptionColor {
    White,
    Green,
    Blue,
    Cyan,
    Red,
    Yellow,
    Magenta,
}

/// Text buffer for caption display.
#[derive(Clone, Debug)]
struct TextBuffer {
    rows: Vec<TextRow>,
}

/// A single row of caption text.
#[derive(Clone, Debug)]
struct TextRow {
    chars: Vec<CaptionChar>,
}

/// A character with styling.
#[derive(Clone, Debug)]
struct CaptionChar {
    ch: char,
    #[allow(dead_code)]
    color: CaptionColor,
    italic: bool,
    underline: bool,
}

impl Cea608Decoder {
    /// Create a new CEA-608 decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mode: DisplayMode::PopOn,
            display_buffer: TextBuffer::new(),
            non_display_buffer: TextBuffer::new(),
            rollup_buffer: TextBuffer::new(),
            rollup_rows: 3,
            cursor_row: 14,
            cursor_col: 0,
            current_color: CaptionColor::White,
            current_italic: false,
            current_underline: false,
            subtitles: Vec::new(),
            current_start_time: None,
            last_timestamp: 0,
        }
    }

    /// Decode a CEA-608 byte pair at the given timestamp.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    pub fn decode(&mut self, byte1: u8, byte2: u8, timestamp_ms: i64) -> SubtitleResult<()> {
        self.last_timestamp = timestamp_ms;

        // Check for null padding
        if byte1 == 0x80 && byte2 == 0x80 {
            return Ok(());
        }

        // Remove parity bits
        let b1 = byte1 & 0x7F;
        let b2 = byte2 & 0x7F;

        // Check for control codes
        if (0x10..=0x1F).contains(&b1) {
            self.decode_control_code(b1, b2, timestamp_ms)?;
        }
        // Special characters
        else if (0x11..=0x13).contains(&b1) && (0x30..=0x3F).contains(&b2) {
            self.decode_special_char(b1, b2)?;
        }
        // Extended characters
        else if (b1 == 0x12 || b1 == 0x1A) && (0x20..=0x3F).contains(&b2) {
            self.decode_extended_char(b1, b2)?;
        }
        // Standard characters
        else if b1 >= 0x20 && b2 >= 0x20 {
            self.add_char(map_cea608_char(b1))?;
            self.add_char(map_cea608_char(b2))?;
        }

        Ok(())
    }

    /// Decode control code.
    fn decode_control_code(&mut self, b1: u8, b2: u8, timestamp_ms: i64) -> SubtitleResult<()> {
        // Check miscellaneous control codes FIRST (before PAC)
        // because 0x14 and 0x1C can be either miscellaneous codes or PAC depending on b2
        let misc_handled = match (b1, b2) {
            // Miscellaneous control codes
            // Resume caption loading (pop-on mode)
            (0x14, 0x20) | (0x1C, 0x20) => {
                self.mode = DisplayMode::PopOn;
                if self.current_start_time.is_none() {
                    self.current_start_time = Some(timestamp_ms);
                }
                true
            }
            // End of caption (display pop-on caption)
            (0x14, 0x2F) | (0x1C, 0x2F) => {
                if self.mode == DisplayMode::PopOn {
                    // Swap buffers first
                    std::mem::swap(&mut self.display_buffer, &mut self.non_display_buffer);
                    // Then flush the now-visible caption
                    self.flush_caption(timestamp_ms)?;
                    self.non_display_buffer.clear();
                }
                true
            }
            // Roll-up 2 rows
            (0x14, 0x25) | (0x1C, 0x25) => {
                self.mode = DisplayMode::RollUp;
                self.rollup_rows = 2;
                true
            }
            // Roll-up 3 rows
            (0x14, 0x26) | (0x1C, 0x26) => {
                self.mode = DisplayMode::RollUp;
                self.rollup_rows = 3;
                true
            }
            // Roll-up 4 rows
            (0x14, 0x27) | (0x1C, 0x27) => {
                self.mode = DisplayMode::RollUp;
                self.rollup_rows = 4;
                true
            }
            // Carriage return (roll-up mode)
            (0x14, 0x2D) | (0x1C, 0x2D) => {
                if self.mode == DisplayMode::RollUp {
                    self.rollup();
                }
                true
            }
            // Erase displayed memory
            (0x14, 0x2C) | (0x1C, 0x2C) => {
                self.display_buffer.clear();
                true
            }
            // Erase non-displayed memory
            (0x14, 0x2E) | (0x1C, 0x2E) => {
                self.non_display_buffer.clear();
                true
            }
            // Resume direct captioning (paint-on mode)
            (0x14, 0x29) | (0x1C, 0x29) => {
                self.mode = DisplayMode::PaintOn;
                true
            }
            // Backspace
            (0x14, 0x21) | (0x1C, 0x21) => {
                self.backspace();
                true
            }
            _ => false,
        };

        if misc_handled {
            return Ok(());
        }

        // Mid-row codes
        if b1 == 0x11 && (0x20..=0x2F).contains(&b2) {
            self.decode_midrow_code(b2);
            return Ok(());
        }

        // PAC (Preamble Address Code) - sets position and style
        if (0x10..=0x17).contains(&b1) || (0x18..=0x1F).contains(&b1) {
            let row = Self::pac_row(b1);
            self.cursor_row = row;
            self.cursor_col = 0;

            // Decode style from b2
            self.current_underline = (b2 & 0x01) != 0;

            if (b2 & 0x0E) >> 1 == 7 {
                self.current_italic = true;
            } else {
                self.current_italic = false;
                self.current_color = Self::pac_color(b2);
            }

            return Ok(());
        }

        // Remaining miscellaneous control codes handled above
        Ok(())
    }

    /// Decode special character.
    fn decode_special_char(&mut self, b1: u8, b2: u8) -> SubtitleResult<()> {
        let ch = match (b1, b2) {
            (0x11, 0x30) => '®',
            (0x11, 0x31) => '°',
            (0x11, 0x32) => '½',
            (0x11, 0x33) => '¿',
            (0x11, 0x34) => '™',
            (0x11, 0x35) => '¢',
            (0x11, 0x36) => '£',
            (0x11, 0x37) => '♪',
            (0x11, 0x38) => 'à',
            (0x11, 0x39) => ' ',
            (0x11, 0x3A) => 'è',
            (0x11, 0x3B) => 'â',
            (0x11, 0x3C) => 'ê',
            (0x11, 0x3D) => 'î',
            (0x11, 0x3E) => 'ô',
            (0x11, 0x3F) => 'û',
            _ => return Ok(()),
        };

        self.add_char(ch)
    }

    /// Decode extended character.
    fn decode_extended_char(&mut self, b1: u8, b2: u8) -> SubtitleResult<()> {
        let ch = match (b1, b2) {
            (0x12, 0x20) => 'Á',
            (0x12, 0x21) => 'É',
            (0x12, 0x22) => 'Ó',
            (0x12, 0x23) => 'Ú',
            (0x12, 0x24) => 'Ü',
            (0x12, 0x25) => 'ü',
            (0x12, 0x26) => '\u{2018}',
            (0x12, 0x27) => '¡',
            (0x12, 0x28) => '*',
            (0x12, 0x29) => '\u{2019}',
            (0x12, 0x2A) => '—',
            (0x12, 0x2B) => '©',
            (0x12, 0x2C) => '℠',
            (0x12, 0x2D) => '•',
            (0x12, 0x2E) => '"',
            (0x12, 0x2F) => '"',
            _ => return Ok(()),
        };

        // Extended characters replace previous character
        self.backspace();
        self.add_char(ch)
    }

    /// Decode mid-row code.
    fn decode_midrow_code(&mut self, b2: u8) {
        self.current_underline = (b2 & 0x01) != 0;

        let style = (b2 & 0x0E) >> 1;
        if style == 7 {
            self.current_italic = true;
        } else {
            self.current_italic = false;
            self.current_color = match style {
                0 => CaptionColor::White,
                1 => CaptionColor::Green,
                2 => CaptionColor::Blue,
                3 => CaptionColor::Cyan,
                4 => CaptionColor::Red,
                5 => CaptionColor::Yellow,
                6 => CaptionColor::Magenta,
                _ => CaptionColor::White,
            };
        }
    }

    /// Add a character to the current buffer.
    fn add_char(&mut self, ch: char) -> SubtitleResult<()> {
        let caption_char = CaptionChar {
            ch,
            color: self.current_color,
            italic: self.current_italic,
            underline: self.current_underline,
        };

        let buffer = match self.mode {
            DisplayMode::PopOn => &mut self.non_display_buffer,
            DisplayMode::RollUp => &mut self.rollup_buffer,
            DisplayMode::PaintOn => &mut self.display_buffer,
        };

        buffer.set_char(self.cursor_row, self.cursor_col, caption_char);
        self.cursor_col += 1;

        // Handle line wrap
        if self.cursor_col >= 32 {
            self.cursor_col = 0;
            if self.cursor_row < 14 {
                self.cursor_row += 1;
            }
        }

        Ok(())
    }

    /// Backspace one character.
    fn backspace(&mut self) {
        if self.cursor_col > 0 {
            self.cursor_col -= 1;
            let buffer = match self.mode {
                DisplayMode::PopOn => &mut self.non_display_buffer,
                DisplayMode::RollUp => &mut self.rollup_buffer,
                DisplayMode::PaintOn => &mut self.display_buffer,
            };
            buffer.clear_char(self.cursor_row, self.cursor_col);
        }
    }

    /// Roll up the text (scroll).
    fn rollup(&mut self) {
        // Move rows up
        for row in 0..14 {
            if row + 1 < 15 {
                self.rollup_buffer.rows[row] = self.rollup_buffer.rows[row + 1].clone();
            }
        }
        // Clear bottom row
        self.rollup_buffer.rows[14].clear();
    }

    /// Flush current caption to subtitle list.
    fn flush_caption(&mut self, end_time: i64) -> SubtitleResult<()> {
        if let Some(start_time) = self.current_start_time {
            let text = self.display_buffer.to_string();
            if !text.trim().is_empty() {
                self.subtitles
                    .push(Subtitle::new(start_time, end_time, text));
            }
        }
        self.current_start_time = None;
        Ok(())
    }

    /// Get row number from PAC byte.
    fn pac_row(b1: u8) -> usize {
        let base = if b1 & 0x08 != 0 { 8 } else { 0 };
        let offset = match b1 & 0x07 {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 4,
            5 => 5,
            6 => 6,
            7 => 7,
            _ => 0,
        };
        base + offset
    }

    /// Get color from PAC byte.
    fn pac_color(b2: u8) -> CaptionColor {
        match (b2 & 0x0E) >> 1 {
            0 => CaptionColor::White,
            1 => CaptionColor::Green,
            2 => CaptionColor::Blue,
            3 => CaptionColor::Cyan,
            4 => CaptionColor::Red,
            5 => CaptionColor::Yellow,
            6 => CaptionColor::Magenta,
            _ => CaptionColor::White,
        }
    }

    /// Get all decoded subtitles.
    #[must_use]
    pub fn take_subtitles(&mut self) -> Vec<Subtitle> {
        std::mem::take(&mut self.subtitles)
    }

    /// Finalize decoding and return all subtitles.
    ///
    /// # Errors
    ///
    /// Returns error if finalization fails.
    pub fn finalize(mut self) -> SubtitleResult<Vec<Subtitle>> {
        // Flush any pending caption
        if self.current_start_time.is_some() {
            self.flush_caption(self.last_timestamp)?;
        }
        Ok(self.subtitles)
    }
}

impl Default for Cea608Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TextBuffer {
    fn new() -> Self {
        Self {
            rows: vec![TextRow::new(); 15],
        }
    }

    fn clear(&mut self) {
        for row in &mut self.rows {
            row.clear();
        }
    }

    fn set_char(&mut self, row: usize, col: usize, ch: CaptionChar) {
        if row < 15 && col < 32 {
            self.rows[row].set_char(col, ch);
        }
    }

    fn clear_char(&mut self, row: usize, col: usize) {
        if row < 15 && col < 32 {
            self.rows[row].clear_char(col);
        }
    }

    fn content(&self) -> String {
        let mut result = String::new();
        for row in &self.rows {
            let row_text = row.content();
            if !row_text.trim().is_empty() {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(&row_text);
            }
        }
        result
    }
}

impl std::fmt::Display for TextBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content())
    }
}

impl TextRow {
    fn new() -> Self {
        Self { chars: Vec::new() }
    }

    fn clear(&mut self) {
        self.chars.clear();
    }

    fn set_char(&mut self, col: usize, ch: CaptionChar) {
        // Expand to fit
        while self.chars.len() <= col {
            self.chars.push(CaptionChar {
                ch: ' ',
                color: CaptionColor::White,
                italic: false,
                underline: false,
            });
        }
        self.chars[col] = ch;
    }

    fn clear_char(&mut self, col: usize) {
        if col < self.chars.len() {
            self.chars[col] = CaptionChar {
                ch: ' ',
                color: CaptionColor::White,
                italic: false,
                underline: false,
            };
        }
    }

    fn content(&self) -> String {
        self.chars.iter().map(|c| c.ch).collect::<String>()
    }
}

impl std::fmt::Display for TextRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content())
    }
}

/// Map CEA-608 character to Unicode.
fn map_cea608_char(byte: u8) -> char {
    match byte {
        0x20..=0x7F => byte as char,
        _ => ' ',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_character_decoding() {
        let mut decoder = Cea608Decoder::new();

        // Resume caption loading
        decoder
            .decode(0x14, 0x20, 0)
            .expect("should succeed in test");

        // Add "HI" (0x48 0x49)
        decoder
            .decode(0x48, 0x49, 100)
            .expect("should succeed in test");

        // End of caption
        decoder
            .decode(0x14, 0x2F, 1000)
            .expect("should succeed in test");

        let subs = decoder.finalize().expect("should succeed in test");
        assert_eq!(subs.len(), 1);
        assert!(subs[0].text.contains("HI"));
    }

    #[test]
    fn test_pac_row_calculation() {
        assert_eq!(Cea608Decoder::pac_row(0x10), 0);
        assert_eq!(Cea608Decoder::pac_row(0x18), 8);
        assert_eq!(Cea608Decoder::pac_row(0x17), 7);
    }
}
