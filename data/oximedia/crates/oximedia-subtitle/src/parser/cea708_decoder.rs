//! CEA-708 closed caption decoder.
//!
//! This module decodes CEA-708 digital television captions from
//! Caption Distribution Packets (CDP) or service blocks.

use crate::{Subtitle, SubtitleError, SubtitleResult};
use std::collections::HashMap;

/// CEA-708 decoder.
pub struct Cea708Decoder {
    /// Active service blocks (1-63).
    services: HashMap<u8, ServiceDecoder>,
    /// Accumulated subtitles.
    subtitles: Vec<Subtitle>,
    /// Last timestamp.
    last_timestamp: i64,
}

/// Decoder for a single CEA-708 service.
struct ServiceDecoder {
    /// Service number (1-63).
    service_number: u8,
    /// Windows (0-7).
    windows: HashMap<u8, Window>,
    /// Current window.
    current_window: Option<u8>,
    /// Pen attributes.
    pen_attributes: PenAttributes,
    /// Pen color.
    pen_color: PenColor,
    /// Subtitle start time.
    start_time: Option<i64>,
}

/// Caption window.
#[derive(Clone, Debug)]
struct Window {
    /// Window ID (0-7).
    id: u8,
    /// Visible flag.
    visible: bool,
    /// Anchor point.
    anchor_vertical: u8,
    anchor_horizontal: u8,
    /// Relative position.
    anchor_relative: bool,
    /// Row count.
    row_count: u8,
    /// Column count.
    column_count: u8,
    /// Text buffer.
    buffer: Vec<Vec<CaptionChar>>,
    /// Current row.
    pen_row: usize,
    /// Current column.
    pen_column: usize,
}

/// Character with styling.
#[derive(Clone, Debug)]
struct CaptionChar {
    ch: char,
    #[allow(dead_code)]
    attributes: PenAttributes,
    #[allow(dead_code)]
    color: PenColor,
}

/// Pen attributes.
#[derive(Clone, Copy, Debug, Default)]
struct PenAttributes {
    font_tag: u8,
    text_tag: u8,
    offset: u8,
    italic: bool,
    underline: bool,
    edge_type: u8,
}

/// Pen color (foreground/background/edge).
#[derive(Clone, Copy, Debug)]
struct PenColor {
    foreground_opacity: u8,
    foreground_r: u8,
    foreground_g: u8,
    foreground_b: u8,
    background_opacity: u8,
    background_r: u8,
    background_g: u8,
    background_b: u8,
}

impl Default for PenColor {
    fn default() -> Self {
        Self {
            foreground_opacity: 3,
            foreground_r: 3,
            foreground_g: 3,
            foreground_b: 3,
            background_opacity: 3,
            background_r: 0,
            background_g: 0,
            background_b: 0,
        }
    }
}

impl Cea708Decoder {
    /// Create a new CEA-708 decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            services: HashMap::new(),
            subtitles: Vec::new(),
            last_timestamp: 0,
        }
    }

    /// Decode a CDP (Caption Distribution Packet) at the given timestamp.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    pub fn decode_cdp(&mut self, data: &[u8], timestamp_ms: i64) -> SubtitleResult<()> {
        self.last_timestamp = timestamp_ms;

        if data.len() < 11 {
            return Err(SubtitleError::ParseError(
                "CDP packet too short".to_string(),
            ));
        }

        // Verify CDP identifier
        if data[0] != 0x96 {
            return Err(SubtitleError::ParseError(
                "Invalid CDP identifier".to_string(),
            ));
        }

        let _cdp_length = data[1];
        let _framerate = (data[2] >> 4) & 0x0F;
        let _flags = data[2] & 0x0F;

        // Find service info section (0x72)
        let mut pos = 3;
        while pos < data.len() {
            let section_type = data[pos];
            pos += 1;

            if section_type == 0x72 {
                // cc_data section
                let cc_count = data[pos] & 0x1F;
                pos += 1;

                for _ in 0..cc_count {
                    if pos + 2 >= data.len() {
                        break;
                    }

                    let cc_valid = (data[pos] & 0x04) != 0;
                    let cc_type = data[pos] & 0x03;
                    let cc_data1 = data[pos + 1];
                    let cc_data2 = data[pos + 2];
                    pos += 3;

                    if cc_valid && cc_type == 0x03 {
                        // CEA-708 data
                        self.decode_service_block(&[cc_data1, cc_data2], timestamp_ms)?;
                    }
                }
            } else {
                // Skip unknown section
                if pos < data.len() {
                    let section_length = data[pos];
                    pos += 1 + section_length as usize;
                }
            }
        }

        Ok(())
    }

    /// Decode service block data.
    fn decode_service_block(&mut self, data: &[u8], timestamp_ms: i64) -> SubtitleResult<()> {
        let mut pos = 0;

        while pos < data.len() {
            let byte = data[pos];
            pos += 1;

            // Control codes (C0: 0x00-0x1F, C1: 0x80-0x9F)
            if byte < 0x20 || (0x80..0xA0).contains(&byte) {
                pos = self.decode_command(data, pos - 1, timestamp_ms)?;
            }
            // G0 characters (0x20-0x7F)
            else if (0x20..0x80).contains(&byte) {
                self.add_char(byte as char)?;
            }
            // G2 characters (0xA0-0xFF) - extended
            else {
                self.add_char(Self::map_g2_char(byte))?;
            }
        }

        Ok(())
    }

    /// Decode a command.
    fn decode_command(
        &mut self,
        data: &[u8],
        pos: usize,
        timestamp_ms: i64,
    ) -> SubtitleResult<usize> {
        let cmd = data[pos];
        let mut new_pos = pos + 1;

        match cmd {
            // NUL - No operation
            0x00 => {}
            // ETX - End of text
            0x03 => {
                // Flush current subtitle
                self.flush_subtitles(timestamp_ms)?;
            }
            // BS - Backspace
            0x08 => {
                self.backspace();
            }
            // FF - Form feed (clear window)
            0x0C => {
                self.clear_current_window();
            }
            // CR - Carriage return
            0x0D => {
                self.carriage_return();
            }
            // CW0-CW7 - Set current window (0x80-0x87)
            0x80..=0x87 => {
                let window_id = cmd & 0x07;
                self.set_current_window(window_id);
            }
            // CLW - Clear windows (0x88)
            0x88 => {
                if new_pos < data.len() {
                    let windows_bitmap = data[new_pos];
                    new_pos += 1;
                    self.clear_windows(windows_bitmap);
                }
            }
            // DSW - Display windows (0x89)
            0x89 => {
                if new_pos < data.len() {
                    let windows_bitmap = data[new_pos];
                    new_pos += 1;
                    self.display_windows(windows_bitmap);
                }
            }
            // HDW - Hide windows (0x8A)
            0x8A => {
                if new_pos < data.len() {
                    let windows_bitmap = data[new_pos];
                    new_pos += 1;
                    self.hide_windows(windows_bitmap);
                }
            }
            // TGW - Toggle windows (0x8B)
            0x8B => {
                if new_pos < data.len() {
                    let windows_bitmap = data[new_pos];
                    new_pos += 1;
                    self.toggle_windows(windows_bitmap);
                }
            }
            // DLW - Delete windows (0x8C)
            0x8C => {
                if new_pos < data.len() {
                    let windows_bitmap = data[new_pos];
                    new_pos += 1;
                    self.delete_windows(windows_bitmap);
                }
            }
            // DLY - Delay (0x8D)
            0x8D => {
                if new_pos < data.len() {
                    let _delay_time = data[new_pos];
                    new_pos += 1;
                }
            }
            // DLC - Delay cancel (0x8E)
            0x8E => {}
            // RST - Reset (0x8F)
            0x8F => {
                self.reset();
            }
            // SPA - Set pen attributes (0x90)
            0x90 => {
                if new_pos + 1 < data.len() {
                    let attr1 = data[new_pos];
                    let attr2 = data[new_pos + 1];
                    new_pos += 2;
                    self.set_pen_attributes(attr1, attr2);
                }
            }
            // SPC - Set pen color (0x91)
            0x91 => {
                if new_pos + 2 < data.len() {
                    let color1 = data[new_pos];
                    let color2 = data[new_pos + 1];
                    let color3 = data[new_pos + 2];
                    new_pos += 3;
                    self.set_pen_color(color1, color2, color3);
                }
            }
            // SPL - Set pen location (0x92)
            0x92 => {
                if new_pos + 1 < data.len() {
                    let row = data[new_pos];
                    let column = data[new_pos + 1];
                    new_pos += 2;
                    self.set_pen_location(row, column);
                }
            }
            // SWA - Set window attributes (0x97)
            0x97 => {
                if new_pos + 3 < data.len() {
                    new_pos += 4;
                    // Window attributes - skip for now
                }
            }
            // DF0-DF7 - Define window (0x98-0x9F)
            0x98..=0x9F => {
                if new_pos + 5 < data.len() {
                    let window_id = cmd & 0x07;
                    self.define_window(window_id, &data[new_pos..new_pos + 6], timestamp_ms)?;
                    new_pos += 6;
                }
            }
            _ => {}
        }

        Ok(new_pos)
    }

    /// Define a window.
    fn define_window(
        &mut self,
        window_id: u8,
        data: &[u8],
        timestamp_ms: i64,
    ) -> SubtitleResult<()> {
        let _priority = (data[0] >> 4) & 0x07;
        let _anchor_relative = (data[0] & 0x08) != 0;
        let anchor_vertical = data[0] & 0x7F;
        let anchor_horizontal = data[1];
        let _anchor_point = (data[2] >> 4) & 0x0F;
        let row_count = data[2] & 0x0F;
        let _column_lock = (data[3] & 0x80) != 0;
        let _row_lock = (data[3] & 0x40) != 0;
        let column_count = data[3] & 0x3F;

        // Get or create service for service 1 (default)
        let service = self.services.entry(1).or_insert_with(|| {
            let mut s = ServiceDecoder::new(1);
            s.start_time = Some(timestamp_ms);
            s
        });

        let window = Window {
            id: window_id,
            visible: false,
            anchor_vertical,
            anchor_horizontal,
            anchor_relative: true,
            row_count: row_count.max(1),
            column_count: column_count.max(1),
            buffer: vec![Vec::new(); row_count.max(1) as usize],
            pen_row: 0,
            pen_column: 0,
        };

        service.windows.insert(window_id, window);
        Ok(())
    }

    /// Set current window.
    fn set_current_window(&mut self, window_id: u8) {
        if let Some(service) = self.services.get_mut(&1) {
            service.current_window = Some(window_id);
        }
    }

    /// Clear windows specified by bitmap.
    fn clear_windows(&mut self, bitmap: u8) {
        if let Some(service) = self.services.get_mut(&1) {
            for window_id in 0..8 {
                if (bitmap & (1 << window_id)) != 0 {
                    if let Some(window) = service.windows.get_mut(&window_id) {
                        window.buffer.clear();
                        window.buffer = vec![Vec::new(); window.row_count as usize];
                        window.pen_row = 0;
                        window.pen_column = 0;
                    }
                }
            }
        }
    }

    /// Display windows specified by bitmap.
    fn display_windows(&mut self, bitmap: u8) {
        if let Some(service) = self.services.get_mut(&1) {
            for window_id in 0..8 {
                if (bitmap & (1 << window_id)) != 0 {
                    if let Some(window) = service.windows.get_mut(&window_id) {
                        window.visible = true;
                    }
                }
            }
        }
    }

    /// Hide windows specified by bitmap.
    fn hide_windows(&mut self, bitmap: u8) {
        if let Some(service) = self.services.get_mut(&1) {
            for window_id in 0..8 {
                if (bitmap & (1 << window_id)) != 0 {
                    if let Some(window) = service.windows.get_mut(&window_id) {
                        window.visible = false;
                    }
                }
            }
        }
    }

    /// Toggle windows specified by bitmap.
    fn toggle_windows(&mut self, bitmap: u8) {
        if let Some(service) = self.services.get_mut(&1) {
            for window_id in 0..8 {
                if (bitmap & (1 << window_id)) != 0 {
                    if let Some(window) = service.windows.get_mut(&window_id) {
                        window.visible = !window.visible;
                    }
                }
            }
        }
    }

    /// Delete windows specified by bitmap.
    fn delete_windows(&mut self, bitmap: u8) {
        if let Some(service) = self.services.get_mut(&1) {
            for window_id in 0..8 {
                if (bitmap & (1 << window_id)) != 0 {
                    service.windows.remove(&window_id);
                }
            }
        }
    }

    /// Reset decoder state.
    fn reset(&mut self) {
        self.services.clear();
    }

    /// Set pen attributes.
    fn set_pen_attributes(&mut self, attr1: u8, attr2: u8) {
        if let Some(service) = self.services.get_mut(&1) {
            service.pen_attributes.font_tag = (attr1 >> 4) & 0x07;
            service.pen_attributes.text_tag = attr1 & 0x0F;
            service.pen_attributes.offset = (attr2 >> 6) & 0x03;
            service.pen_attributes.italic = (attr2 & 0x02) != 0;
            service.pen_attributes.underline = (attr2 & 0x01) != 0;
            service.pen_attributes.edge_type = (attr2 >> 2) & 0x07;
        }
    }

    /// Set pen color.
    fn set_pen_color(&mut self, color1: u8, color2: u8, color3: u8) {
        if let Some(service) = self.services.get_mut(&1) {
            service.pen_color.foreground_opacity = (color1 >> 6) & 0x03;
            service.pen_color.foreground_r = (color1 >> 4) & 0x03;
            service.pen_color.foreground_g = (color1 >> 2) & 0x03;
            service.pen_color.foreground_b = color1 & 0x03;
            service.pen_color.background_opacity = (color2 >> 6) & 0x03;
            service.pen_color.background_r = (color2 >> 4) & 0x03;
            service.pen_color.background_g = (color2 >> 2) & 0x03;
            service.pen_color.background_b = color2 & 0x03;
        }
    }

    /// Set pen location.
    fn set_pen_location(&mut self, row: u8, column: u8) {
        if let Some(service) = self.services.get_mut(&1) {
            if let Some(window_id) = service.current_window {
                if let Some(window) = service.windows.get_mut(&window_id) {
                    window.pen_row = (row as usize).min(window.row_count as usize - 1);
                    window.pen_column = (column as usize).min(window.column_count as usize - 1);
                }
            }
        }
    }

    /// Add a character to the current window.
    fn add_char(&mut self, ch: char) -> SubtitleResult<()> {
        if let Some(service) = self.services.get_mut(&1) {
            if let Some(window_id) = service.current_window {
                if let Some(window) = service.windows.get_mut(&window_id) {
                    let caption_char = CaptionChar {
                        ch,
                        attributes: service.pen_attributes,
                        color: service.pen_color,
                    };

                    // Ensure row exists
                    while window.buffer.len() <= window.pen_row {
                        window.buffer.push(Vec::new());
                    }

                    // Ensure column exists
                    while window.buffer[window.pen_row].len() <= window.pen_column {
                        window.buffer[window.pen_row].push(CaptionChar {
                            ch: ' ',
                            attributes: service.pen_attributes,
                            color: service.pen_color,
                        });
                    }

                    window.buffer[window.pen_row][window.pen_column] = caption_char;
                    window.pen_column += 1;
                }
            }
        }
        Ok(())
    }

    /// Backspace.
    fn backspace(&mut self) {
        if let Some(service) = self.services.get_mut(&1) {
            if let Some(window_id) = service.current_window {
                if let Some(window) = service.windows.get_mut(&window_id) {
                    if window.pen_column > 0 {
                        window.pen_column -= 1;
                        if window.pen_row < window.buffer.len()
                            && window.pen_column < window.buffer[window.pen_row].len()
                        {
                            window.buffer[window.pen_row][window.pen_column].ch = ' ';
                        }
                    }
                }
            }
        }
    }

    /// Carriage return.
    fn carriage_return(&mut self) {
        if let Some(service) = self.services.get_mut(&1) {
            if let Some(window_id) = service.current_window {
                if let Some(window) = service.windows.get_mut(&window_id) {
                    window.pen_column = 0;
                    if window.pen_row + 1 < window.row_count as usize {
                        window.pen_row += 1;
                    }
                }
            }
        }
    }

    /// Clear current window.
    fn clear_current_window(&mut self) {
        if let Some(service) = self.services.get_mut(&1) {
            if let Some(window_id) = service.current_window {
                if let Some(window) = service.windows.get_mut(&window_id) {
                    window.buffer.clear();
                    window.buffer = vec![Vec::new(); window.row_count as usize];
                    window.pen_row = 0;
                    window.pen_column = 0;
                }
            }
        }
    }

    /// Flush visible subtitles.
    fn flush_subtitles(&mut self, end_time: i64) -> SubtitleResult<()> {
        for service in self.services.values() {
            for window in service.windows.values() {
                if window.visible {
                    let text = window.to_string();
                    if !text.trim().is_empty() {
                        if let Some(start_time) = service.start_time {
                            self.subtitles
                                .push(Subtitle::new(start_time, end_time, text));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Map G2 character to Unicode.
    fn map_g2_char(byte: u8) -> char {
        match byte {
            0xA0 => ' ',
            0xA1 => '¡',
            0xA2 => '¢',
            0xA3 => '£',
            0xA4 => '¤',
            0xA5 => '¥',
            0xA6 => '¦',
            0xA7 => '§',
            0xA8 => '¨',
            0xA9 => '©',
            0xAA => 'ª',
            0xAB => '«',
            0xAC => '¬',
            0xAD => '\u{AD}',
            0xAE => '®',
            0xAF => '¯',
            _ => byte as char,
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
        self.flush_subtitles(self.last_timestamp)?;
        Ok(self.subtitles)
    }
}

impl Default for Cea708Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ServiceDecoder {
    fn new(service_number: u8) -> Self {
        Self {
            service_number,
            windows: HashMap::new(),
            current_window: None,
            pen_attributes: PenAttributes::default(),
            pen_color: PenColor::default(),
            start_time: None,
        }
    }
}

impl Window {
    fn content(&self) -> String {
        let mut result = String::new();
        for row in &self.buffer {
            let row_text: String = row.iter().map(|c| c.ch).collect();
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

impl std::fmt::Display for Window {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = Cea708Decoder::new();
        assert_eq!(decoder.services.len(), 0);
    }

    #[test]
    fn test_window_definition() {
        let mut decoder = Cea708Decoder::new();
        let window_data = [0x00, 0x00, 0x44, 0x20, 0x00, 0x00];
        decoder
            .define_window(0, &window_data, 0)
            .expect("should succeed in test");

        let service = decoder.services.get(&1).expect("should succeed in test");
        assert!(service.windows.contains_key(&0));
    }
}
