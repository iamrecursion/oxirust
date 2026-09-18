//! DVB Teletext subtitle extraction per ETS 300 706.
//!
//! Provides `TeletextParser` that decodes teletext pages from data packets,
//! extracting subtitle text from teletext-based subtitle services.

use crate::{SubtitleError, SubtitleResult};

/// A decoded teletext page.
#[derive(Clone, Debug)]
pub struct TeletextPage {
    /// Magazine number (0-7).
    pub magazine: u8,
    /// Page number (0x00-0xFF).
    pub page_number: u8,
    /// Subpage code.
    pub subpage: u16,
    /// Decoded text lines (up to 25 rows, 40 columns each).
    pub lines: Vec<String>,
    /// Whether this page contains subtitle data.
    pub is_subtitle: bool,
    /// Presentation timestamp in milliseconds (if available).
    pub pts_ms: Option<i64>,
}

impl TeletextPage {
    /// Create a new empty teletext page.
    #[must_use]
    pub fn new(magazine: u8, page_number: u8) -> Self {
        Self {
            magazine,
            page_number,
            subpage: 0,
            lines: Vec::new(),
            is_subtitle: false,
            pts_ms: None,
        }
    }

    /// Get the full page number as a 3-digit hex value (e.g., 0x888 = page 888).
    #[must_use]
    pub fn full_page_number(&self) -> u16 {
        u16::from(self.magazine) * 256 + u16::from(self.page_number)
    }

    /// Concatenate non-empty lines into subtitle text.
    #[must_use]
    pub fn subtitle_text(&self) -> String {
        self.lines
            .iter()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Hamming 8/4 decoding table for teletext.
///
/// Returns the decoded 4-bit value, or `None` for uncorrectable errors.
fn hamming_8_4_decode(byte: u8) -> Option<u8> {
    // ETS 300 706 Table 1: Hamming 8/4 decoding
    static HAMMING_TABLE: [i8; 256] = {
        let mut table = [-1i8; 256];
        // Build from the known encoded values
        // For each 4-bit value d3d2d1d0, encode using hamming 8/4
        // P1=d1^d3^d4, P2=d1^d2^d4, D1=d1, P3=d2^d3^d4, D2=d2, D3=d3, P4=... D4=d4
        // Using a simplified approach: pre-computed decoding
        let encoded: [(u8, u8); 16] = [
            (0x15, 0),
            (0x02, 1),
            (0x49, 2),
            (0x5E, 3),
            (0x64, 4),
            (0x73, 5),
            (0x38, 6),
            (0x2F, 7),
            (0xD0, 8),
            (0xC7, 9),
            (0x8C, 10),
            (0x9B, 11),
            (0xA1, 12),
            (0xB6, 13),
            (0xFD, 14),
            (0xEA, 15),
        ];
        let mut i = 0;
        while i < 16 {
            table[encoded[i].0 as usize] = encoded[i].1 as i8;
            i += 1;
        }
        table
    };

    let val = HAMMING_TABLE[byte as usize];
    if val < 0 {
        None
    } else {
        Some(val as u8)
    }
}

/// Odd parity check for teletext character bytes.
fn odd_parity_strip(byte: u8) -> u8 {
    // Strip the parity bit (bit 7) and return the 7-bit character
    byte & 0x7F
}

/// Teletext character set mapping (G0 Latin, default English).
///
/// Maps 7-bit teletext codes to Unicode characters.
fn teletext_to_char(code: u8) -> char {
    // ETS 300 706 Table 36: G0 Latin National Option Sub-sets
    // Basic ASCII range is mostly identical
    match code {
        0x00..=0x1F => ' ', // Control codes -> space
        0x20..=0x7E => code as char,
        0x7F => '\u{25A0}', // Block character
        _ => ' ',
    }
}

/// Teletext packet types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeletextPacketType {
    /// Page header (row 0).
    PageHeader,
    /// Normal data row (rows 1-24).
    DataRow,
    /// Page enhancement data packet.
    Enhancement,
    /// Unknown packet type.
    Unknown,
}

/// A parsed teletext data unit.
#[derive(Clone, Debug)]
pub struct TeletextPacket {
    /// Magazine number (0-7).
    pub magazine: u8,
    /// Packet number (row, 0-31).
    pub row: u8,
    /// Packet type classification.
    pub packet_type: TeletextPacketType,
    /// Raw data bytes (40 bytes for rows 0-24).
    pub data: Vec<u8>,
}

/// Teletext subtitle parser.
///
/// Decodes teletext pages from PES data packets per ETS 300 706
/// and extracts subtitle text.
#[derive(Clone, Debug)]
pub struct TeletextParser {
    /// Target page numbers to extract (e.g., page 888 for UK subtitles).
    pub target_pages: Vec<u16>,
    /// Current page buffers indexed by magazine.
    page_buffers: [Option<TeletextPage>; 8],
    /// Completed pages ready for output.
    completed_pages: Vec<TeletextPage>,
    /// Current presentation timestamp.
    current_pts_ms: Option<i64>,
}

impl TeletextParser {
    /// Create a new teletext parser targeting the given page numbers.
    ///
    /// Common subtitle pages: 888 (UK), 777 (many European countries).
    #[must_use]
    pub fn new(target_pages: Vec<u16>) -> Self {
        Self {
            target_pages,
            page_buffers: Default::default(),
            completed_pages: Vec::new(),
            current_pts_ms: None,
        }
    }

    /// Create a parser for common subtitle page 888.
    #[must_use]
    pub fn default_subtitle_parser() -> Self {
        Self::new(vec![0x888])
    }

    /// Set the current PTS for incoming data.
    pub fn set_pts(&mut self, pts_ms: i64) {
        self.current_pts_ms = Some(pts_ms);
    }

    /// Feed a teletext data unit (typically 44 bytes per ETS 300 706).
    ///
    /// # Errors
    ///
    /// Returns error if the data is too short or contains uncorrectable errors.
    pub fn feed_data_unit(&mut self, data: &[u8]) -> SubtitleResult<()> {
        if data.len() < 2 {
            return Err(SubtitleError::ParseError(
                "Teletext data unit too short".to_string(),
            ));
        }

        // Data unit header: data_unit_id (1 byte) + data_unit_length (1 byte) + payload
        let data_unit_id = data[0];
        let _data_unit_length = data[1];

        // 0x02 = EBU teletext non-subtitle data
        // 0x03 = EBU teletext subtitle data
        if data_unit_id != 0x02 && data_unit_id != 0x03 {
            return Ok(()); // Not teletext data, skip
        }

        let is_subtitle_stream = data_unit_id == 0x03;

        if data.len() < 44 {
            return Err(SubtitleError::ParseError(
                "Teletext data unit payload too short (need 44 bytes)".to_string(),
            ));
        }

        // Payload starts at byte 2 (after header)
        // Byte 2: field_parity + line_offset (can be ignored for subtitle extraction)
        // Byte 3: framing code (should be 0xE4)
        let framing_code = data[3];
        if framing_code != 0xE4 {
            return Err(SubtitleError::ParseError(format!(
                "Invalid framing code: 0x{framing_code:02X} (expected 0xE4)"
            )));
        }

        // Bytes 4-5: magazine and row address (Hamming 8/4 coded)
        let addr_byte1 = data[4];
        let addr_byte2 = data[5];

        let addr1 = hamming_8_4_decode(addr_byte1);
        let addr2 = hamming_8_4_decode(addr_byte2);

        let (magazine, row) = match (addr1, addr2) {
            (Some(a1), Some(a2)) => {
                let magazine = a1 & 0x07;
                let row = (a1 >> 3) | (a2 << 1);
                // Magazine 0 is actually magazine 8
                let magazine = if magazine == 0 { 8 } else { magazine };
                (magazine, row)
            }
            _ => {
                return Err(SubtitleError::ParseError(
                    "Hamming decode error in address bytes".to_string(),
                ));
            }
        };

        // Decode based on row number
        if row == 0 {
            // Page header
            self.handle_page_header(magazine, &data[6..44], is_subtitle_stream)?;
        } else if row <= 24 {
            // Normal data row
            self.handle_data_row(magazine, row, &data[6..44])?;
        }
        // Rows 25-31 are enhancement data, ignored for basic subtitle extraction

        Ok(())
    }

    /// Handle a page header (row 0).
    fn handle_page_header(
        &mut self,
        magazine: u8,
        payload: &[u8],
        is_subtitle: bool,
    ) -> SubtitleResult<()> {
        if payload.len() < 6 {
            return Err(SubtitleError::ParseError(
                "Page header payload too short".to_string(),
            ));
        }

        // Bytes 0-1: page number (Hamming 8/4)
        let page_units = hamming_8_4_decode(payload[0]);
        let page_tens = hamming_8_4_decode(payload[1]);

        let page_number = match (page_units, page_tens) {
            (Some(u), Some(t)) => t * 16 + u,
            _ => {
                return Err(SubtitleError::ParseError(
                    "Hamming decode error in page number".to_string(),
                ));
            }
        };

        // Bytes 2-5: subpage (Hamming 8/4)
        let sub_bytes: Vec<Option<u8>> = payload[2..6]
            .iter()
            .map(|&b| hamming_8_4_decode(b))
            .collect();

        let subpage = if let (Some(s0), Some(s1), Some(s2), Some(s3)) =
            (sub_bytes[0], sub_bytes[1], sub_bytes[2], sub_bytes[3])
        {
            u16::from(s0) | (u16::from(s1) << 4) | (u16::from(s2) << 8) | (u16::from(s3) << 12)
        } else {
            0
        };

        let mag_idx = (magazine.saturating_sub(1)).min(7) as usize;

        // If there is an existing page buffer for this magazine, complete it
        if let Some(existing) = self.page_buffers[mag_idx].take() {
            let full_page = existing.full_page_number();
            if self.target_pages.is_empty() || self.target_pages.contains(&full_page) {
                self.completed_pages.push(existing);
            }
        }

        // Start new page
        let mut page = TeletextPage::new(magazine, page_number);
        page.subpage = subpage;
        page.is_subtitle = is_subtitle;
        page.pts_ms = self.current_pts_ms;

        // Decode the remaining header text (bytes 6 onwards, typically display row 0)
        if payload.len() > 6 {
            let header_text = decode_teletext_line(&payload[6..]);
            page.lines.push(header_text);
        }

        self.page_buffers[mag_idx] = Some(page);

        Ok(())
    }

    /// Handle a normal data row (rows 1-24).
    fn handle_data_row(&mut self, magazine: u8, row: u8, payload: &[u8]) -> SubtitleResult<()> {
        let mag_idx = (magazine.saturating_sub(1)).min(7) as usize;

        if let Some(page) = self.page_buffers[mag_idx].as_mut() {
            // Pad with empty lines if needed (rows might arrive out of order)
            let row_idx = row as usize;
            while page.lines.len() < row_idx {
                page.lines.push(String::new());
            }

            let line = decode_teletext_line(payload);

            if row_idx < page.lines.len() {
                page.lines[row_idx] = line;
            } else {
                page.lines.push(line);
            }
        }

        Ok(())
    }

    /// Flush all buffered pages and return completed subtitle pages.
    pub fn flush(&mut self) -> Vec<TeletextPage> {
        // Move any buffered pages to completed
        for buf in self.page_buffers.iter_mut() {
            if let Some(page) = buf.take() {
                let full_page = page.full_page_number();
                if self.target_pages.is_empty() || self.target_pages.contains(&full_page) {
                    self.completed_pages.push(page);
                }
            }
        }
        std::mem::take(&mut self.completed_pages)
    }

    /// Drain completed pages without flushing buffers.
    pub fn drain_completed(&mut self) -> Vec<TeletextPage> {
        std::mem::take(&mut self.completed_pages)
    }

    /// Check if there are completed pages ready.
    #[must_use]
    pub fn has_completed_pages(&self) -> bool {
        !self.completed_pages.is_empty()
    }
}

/// Decode a teletext data line (up to 40 characters) from raw bytes.
fn decode_teletext_line(data: &[u8]) -> String {
    let mut result = String::with_capacity(40);
    for &byte in data.iter().take(40) {
        let code = odd_parity_strip(byte);
        result.push(teletext_to_char(code));
    }
    // Trim trailing spaces
    result.truncate(result.trim_end().len());
    result
}

/// Build a synthetic teletext data unit for testing purposes.
///
/// Creates a minimal 44-byte data unit with the given parameters.
#[must_use]
pub fn build_test_data_unit(data_unit_id: u8, magazine: u8, row: u8, text: &str) -> Vec<u8> {
    let mut data = vec![0u8; 44];
    data[0] = data_unit_id;
    data[1] = 42; // data_unit_length
    data[2] = 0; // field_parity + line_offset
    data[3] = 0xE4; // framing code

    // Encode magazine and row with Hamming 8/4
    // For testing, use a known encoded value for magazine/row
    // We store the encoded bytes directly from the Hamming table
    let encoded_values: [u8; 16] = [
        0x15, 0x02, 0x49, 0x5E, 0x64, 0x73, 0x38, 0x2F, 0xD0, 0xC7, 0x8C, 0x9B, 0xA1, 0xB6, 0xFD,
        0xEA,
    ];

    // addr1 lower 3 bits = magazine, bit 3 = row bit 0
    let mag_adjusted = if magazine == 8 { 0 } else { magazine };
    let addr1_val = (mag_adjusted & 0x07) | ((row & 0x01) << 3);
    let addr2_val = (row >> 1) & 0x0F;

    if (addr1_val as usize) < encoded_values.len() {
        data[4] = encoded_values[addr1_val as usize];
    }
    if (addr2_val as usize) < encoded_values.len() {
        data[5] = encoded_values[addr2_val as usize];
    }

    // For row 0, we need page number and subpage in bytes 6-11
    if row == 0 {
        // Page number: e.g., 0x88 for page 888
        // Use hamming encoded 0x08 for units, 0x08 for tens
        if encoded_values.len() > 8 {
            data[6] = encoded_values[8]; // page units = 8
            data[7] = encoded_values[8]; // page tens = 8
        }
        // Subpage = 0
        data[8] = encoded_values[0];
        data[9] = encoded_values[0];
        data[10] = encoded_values[0];
        data[11] = encoded_values[0];

        // Text starts at byte 12
        for (i, ch) in text.chars().enumerate() {
            let idx = 12 + i;
            if idx < 44 {
                data[idx] = ch as u8;
            }
        }
    } else {
        // Data row: text in bytes 6-43
        for (i, ch) in text.chars().enumerate() {
            let idx = 6 + i;
            if idx < 44 {
                data[idx] = ch as u8;
            }
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_teletext_page_basic() {
        let mut page = TeletextPage::new(8, 0x88);
        page.lines.push("Hello World".to_string());
        page.lines.push("".to_string());
        page.lines.push("Second line".to_string());
        assert_eq!(page.subtitle_text(), "Hello World\nSecond line");
    }

    #[test]
    fn test_teletext_page_full_number() {
        let page = TeletextPage::new(8, 0x88);
        assert_eq!(page.full_page_number(), 0x888);
    }

    #[test]
    fn test_hamming_decode_known_values() {
        assert_eq!(hamming_8_4_decode(0x15), Some(0));
        assert_eq!(hamming_8_4_decode(0x02), Some(1));
        assert_eq!(hamming_8_4_decode(0x49), Some(2));
        assert_eq!(hamming_8_4_decode(0xEA), Some(15));
    }

    #[test]
    fn test_hamming_decode_invalid() {
        // 0xFF is not in the table
        assert_eq!(hamming_8_4_decode(0xFF), None);
    }

    #[test]
    fn test_odd_parity_strip() {
        assert_eq!(odd_parity_strip(0xC1), 0x41); // 'A' with parity
        assert_eq!(odd_parity_strip(0x41), 0x41); // 'A' without parity
    }

    #[test]
    fn test_teletext_to_char() {
        assert_eq!(teletext_to_char(0x41), 'A');
        assert_eq!(teletext_to_char(0x20), ' ');
        assert_eq!(teletext_to_char(0x00), ' '); // control code
    }

    #[test]
    fn test_decode_teletext_line() {
        let data = b"Hello Teletext";
        let line = decode_teletext_line(data);
        assert_eq!(line, "Hello Teletext");
    }

    #[test]
    fn test_decode_teletext_line_with_parity() {
        // 'H' = 0x48, with parity bit set = 0xC8
        let data = [0xC8, 0x69]; // 'H' with parity, 'i'
        let line = decode_teletext_line(&data);
        assert_eq!(line, "Hi");
    }

    #[test]
    fn test_teletext_parser_create() {
        let parser = TeletextParser::new(vec![0x888]);
        assert_eq!(parser.target_pages, vec![0x888]);
        assert!(!parser.has_completed_pages());
    }

    #[test]
    fn test_teletext_parser_default() {
        let parser = TeletextParser::default_subtitle_parser();
        assert_eq!(parser.target_pages, vec![0x888]);
    }

    #[test]
    fn test_teletext_parser_set_pts() {
        let mut parser = TeletextParser::new(vec![]);
        parser.set_pts(1000);
        assert_eq!(parser.current_pts_ms, Some(1000));
    }

    #[test]
    fn test_teletext_parser_short_data() {
        let mut parser = TeletextParser::new(vec![]);
        let result = parser.feed_data_unit(&[0x03]);
        assert!(result.is_err());
    }

    #[test]
    fn test_teletext_parser_non_teletext_data() {
        let mut parser = TeletextParser::new(vec![]);
        // data_unit_id = 0x01 (not teletext)
        let data = vec![0x01, 42, 0, 0xE4, 0, 0, 0, 0, 0, 0];
        let result = parser.feed_data_unit(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_teletext_parser_bad_framing_code() {
        let mut parser = TeletextParser::new(vec![]);
        let mut data = vec![0u8; 44];
        data[0] = 0x03;
        data[1] = 42;
        data[3] = 0x00; // wrong framing code
        let result = parser.feed_data_unit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_test_data_unit_length() {
        let data = build_test_data_unit(0x03, 8, 0, "Test");
        assert_eq!(data.len(), 44);
        assert_eq!(data[0], 0x03);
        assert_eq!(data[3], 0xE4);
    }

    #[test]
    fn test_teletext_parser_flush_empty() {
        let mut parser = TeletextParser::new(vec![]);
        let pages = parser.flush();
        assert!(pages.is_empty());
    }

    #[test]
    fn test_teletext_page_empty_subtitle_text() {
        let page = TeletextPage::new(1, 0);
        assert_eq!(page.subtitle_text(), "");
    }

    #[test]
    fn test_teletext_page_whitespace_lines() {
        let mut page = TeletextPage::new(1, 0);
        page.lines.push("   ".to_string());
        page.lines.push("Text".to_string());
        page.lines.push("  ".to_string());
        assert_eq!(page.subtitle_text(), "Text");
    }

    #[test]
    fn test_teletext_parser_drain_completed() {
        let mut parser = TeletextParser::new(vec![]);
        let pages = parser.drain_completed();
        assert!(pages.is_empty());
    }
}
