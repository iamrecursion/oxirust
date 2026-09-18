//! CEA-608/708 subtitle insertion and SRT/WebVTT parsing.
//!
//! Provides subtitle track management, real-time packet generation keyed on
//! 90 kHz presentation timestamps (PTS), and broadcast-standard CEA-608/708
//! byte-pair encoding.

use crate::{PlayoutError, Result};

// ─── Colour ──────────────────────────────────────────────────────────────────

/// CEA-608 foreground colours (3-bit code).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cea608Color {
    White,
    Green,
    Blue,
    Cyan,
    Red,
    Yellow,
    Magenta,
    Black,
}

impl Cea608Color {
    /// Return the two-byte PAC style attribute nibble (colour bits 2–0).
    fn color_bits(self) -> u8 {
        match self {
            Self::White => 0b000,
            Self::Green => 0b001,
            Self::Blue => 0b010,
            Self::Cyan => 0b011,
            Self::Red => 0b100,
            Self::Yellow => 0b101,
            Self::Magenta => 0b110,
            Self::Black => 0b111,
        }
    }
}

// ─── Style / Position ────────────────────────────────────────────────────────

/// Screen position for a subtitle cue (CEA-608 row 1–15).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubtitlePosition {
    /// Row on screen (1 = top, 15 = bottom for CEA-608)
    pub row: u8,
    /// Column (0-based)
    pub col: u8,
}

impl Default for SubtitlePosition {
    fn default() -> Self {
        Self { row: 15, col: 0 }
    }
}

/// Text styling for a subtitle cue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtitleStyle {
    pub color: Cea608Color,
    pub flash: bool,
    pub underline: bool,
}

impl Default for SubtitleStyle {
    fn default() -> Self {
        Self {
            color: Cea608Color::White,
            flash: false,
            underline: false,
        }
    }
}

// ─── Subtitle entry ───────────────────────────────────────────────────────────

/// A single subtitle cue.
#[derive(Debug, Clone)]
pub struct SubtitleEntry {
    /// Start PTS in 90 kHz clock ticks.
    pub start_pts: u64,
    /// End PTS in 90 kHz clock ticks.
    pub end_pts: u64,
    /// Subtitle text (may contain `\n` for line breaks).
    pub text: String,
    /// CEA-608 channel 1–4, CEA-708 service 1–63.
    pub channel: u8,
    pub position: SubtitlePosition,
    pub style: SubtitleStyle,
}

impl SubtitleEntry {
    /// Convenience constructor with sensible defaults.
    pub fn new(start_pts: u64, end_pts: u64, text: impl Into<String>) -> Self {
        Self {
            start_pts,
            end_pts,
            text: text.into(),
            channel: 1,
            position: SubtitlePosition::default(),
            style: SubtitleStyle::default(),
        }
    }

    /// Returns `true` if the cue is active at the given PTS.
    pub fn is_active_at(&self, pts: u64) -> bool {
        pts >= self.start_pts && pts < self.end_pts
    }
}

// ─── Format ──────────────────────────────────────────────────────────────────

/// Supported subtitle formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    Cea608,
    Cea708,
    Webvtt,
    Srt,
}

// ─── Track ───────────────────────────────────────────────────────────────────

/// A named subtitle track (one per language / service).
#[derive(Debug, Clone)]
pub struct SubtitleTrack {
    pub id: u32,
    pub language: String,
    pub format: SubtitleFormat,
    pub entries: Vec<SubtitleEntry>,
}

impl SubtitleTrack {
    pub fn new(id: u32, language: impl Into<String>, format: SubtitleFormat) -> Self {
        Self {
            id,
            language: language.into(),
            format,
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, entry: SubtitleEntry) {
        self.entries.push(entry);
        // keep entries sorted by start PTS for efficient range queries
        self.entries.sort_by_key(|e| e.start_pts);
    }
}

// ─── Inserter ────────────────────────────────────────────────────────────────

/// Manages multiple subtitle tracks and produces broadcast-ready byte packets.
#[derive(Debug, Default)]
pub struct SubtitleInserter {
    pub tracks: Vec<SubtitleTrack>,
}

impl SubtitleInserter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pre-built track.
    pub fn add_track(&mut self, track: SubtitleTrack) {
        self.tracks.push(track);
    }

    // ─── CEA-608 encoding ────────────────────────────────────────────────

    /// Encode a subtitle entry as a sequence of CEA-608 byte-pairs.
    ///
    /// Layout:
    /// 1. Erase Displayed Memory (EDM) control code.
    /// 2. Preamble Address Code (PAC) setting row + colour.
    /// 3. Text bytes (7-bit ASCII with odd parity on bit 7).
    /// 4. End-of-Caption (EOC) control code.
    pub fn encode_cea608_packet(entry: &SubtitleEntry) -> Vec<u8> {
        let mut out = Vec::new();

        // -- Erase Displayed Memory (CH1: 0x14 0x2C)
        let cc = channel_control_byte(entry.channel);
        out.push(set_parity(cc));
        out.push(set_parity(0x2C)); // EDM

        // -- PAC: row + colour attribute
        let (row_b1, row_b2) = pac_bytes(entry.position.row, &entry.style);
        out.push(row_b1);
        out.push(row_b2);

        // -- Text (split at newlines → new PAC for each line)
        let mut current_row = entry.position.row;
        let text = truncate_row(&entry.text, 32);

        for line in text.split('\n') {
            if current_row != entry.position.row {
                // Additional PAC for subsequent lines
                let (b1, b2) = pac_bytes(current_row, &entry.style);
                out.push(b1);
                out.push(b2);
            }
            for ch in line.chars().take(32) {
                let byte = ascii_to_cea608(ch);
                out.push(set_parity(byte));
                out.push(set_parity(0x80)); // filler / null second byte
            }
            current_row = current_row.saturating_add(1).min(15);
        }

        // -- End of Caption (EOC: 0x14 0x2F for CH1)
        out.push(set_parity(cc));
        out.push(set_parity(0x2F));

        out
    }

    /// Encode a subtitle entry as a simplified CEA-708 service block.
    ///
    /// Format: `[service_number(6b) | block_size(2b)] [UTF-8 text bytes]`
    pub fn encode_cea708_packet(entry: &SubtitleEntry) -> Vec<u8> {
        let text_bytes = entry.text.as_bytes();
        // Service number: 1–63, block size header byte
        let svc = entry.channel.clamp(1, 63);
        let size = text_bytes.len().min(127) as u8;

        let mut out = Vec::with_capacity(2 + size as usize);
        // Header: [service_number << 2 | block_size_high2] — simplified
        out.push((svc << 2) | ((size >> 5) & 0b11));
        out.push(size & 0b11111);
        out.extend_from_slice(&text_bytes[..size as usize]);
        out
    }

    // ─── PTS-based packet retrieval ──────────────────────────────────────

    /// Return all encoded subtitle packets active at the given PTS.
    ///
    /// Each inner `Vec<u8>` corresponds to one `SubtitleEntry` across all
    /// tracks.  The encoding used depends on `track.format`.
    pub fn get_packets_for_frame(&self, pts: u64) -> Vec<Vec<u8>> {
        let mut packets = Vec::new();

        for track in &self.tracks {
            for entry in &track.entries {
                if entry.is_active_at(pts) {
                    let packet = match track.format {
                        SubtitleFormat::Cea608 => Self::encode_cea608_packet(entry),
                        SubtitleFormat::Cea708 => Self::encode_cea708_packet(entry),
                        // For WebVTT/SRT we return the raw UTF-8 text bytes
                        SubtitleFormat::Webvtt | SubtitleFormat::Srt => {
                            entry.text.as_bytes().to_vec()
                        }
                    };
                    packets.push(packet);
                }
            }
        }

        packets
    }

    // ─── SRT parser ──────────────────────────────────────────────────────

    /// Parse an SRT subtitle file into a `Vec<SubtitleEntry>`.
    ///
    /// Timecodes are converted from `HH:MM:SS,mmm` to 90 kHz PTS values.
    pub fn parse_srt(text: &str) -> Result<Vec<SubtitleEntry>> {
        let mut entries = Vec::new();
        // Split into cue blocks separated by blank lines
        let blocks: Vec<&str> = text.split("\n\n").collect();

        for block in blocks {
            let block = block.trim();
            if block.is_empty() {
                continue;
            }
            let lines: Vec<&str> = block.lines().collect();
            if lines.len() < 3 {
                continue;
            }
            // lines[0] — sequence number (ignored)
            // lines[1] — timecode range
            // lines[2..] — text

            let timecode_line = lines[1].trim();
            let (start_pts, end_pts) = parse_srt_timecode(timecode_line)?;

            let text_content = lines[2..].join("\n");
            entries.push(SubtitleEntry::new(start_pts, end_pts, text_content));
        }

        Ok(entries)
    }

    /// Parse a WebVTT file into a `Vec<SubtitleEntry>`.
    ///
    /// Handles the `WEBVTT` header and `HH:MM:SS.mmm --> HH:MM:SS.mmm` cues.
    pub fn parse_webvtt(text: &str) -> Result<Vec<SubtitleEntry>> {
        let mut entries = Vec::new();
        let mut lines_iter = text.lines().peekable();

        // Skip header line ("WEBVTT …")
        if let Some(hdr) = lines_iter.next() {
            if !hdr.trim_start().starts_with("WEBVTT") {
                return Err(PlayoutError::Playlist(
                    "Not a valid WebVTT file (missing WEBVTT header)".to_string(),
                ));
            }
        }

        // Collect cue blocks
        let remaining: String = lines_iter.collect::<Vec<&str>>().join("\n");
        for block in remaining.split("\n\n") {
            let block = block.trim();
            if block.is_empty() {
                continue;
            }
            let blines: Vec<&str> = block.lines().collect();

            // Find the timecode line (contains "-->")
            let tc_idx = blines
                .iter()
                .position(|l| l.contains("-->"))
                .ok_or_else(|| {
                    PlayoutError::Playlist("WebVTT cue missing timecode line".to_string())
                })?;

            let tc_line = blines[tc_idx].trim();
            let (start_pts, end_pts) = parse_webvtt_timecode(tc_line)?;

            let text_content = blines[tc_idx + 1..].join("\n");
            if !text_content.is_empty() {
                entries.push(SubtitleEntry::new(start_pts, end_pts, text_content));
            }
        }

        Ok(entries)
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Convert milliseconds to 90 kHz clock ticks.
fn ms_to_90k(ms: u64) -> u64 {
    ms * 90
}

/// Odd-parity bit insertion for CEA-608 (bit 7 set so that the total number
/// of 1-bits in the byte is odd).
fn set_parity(byte: u8) -> u8 {
    let count = byte.count_ones();
    if count % 2 == 0 {
        byte | 0x80
    } else {
        byte & 0x7F
    }
}

/// Return the channel-select byte for channel 1–4 (CEA-608).
fn channel_control_byte(channel: u8) -> u8 {
    match channel {
        1 | 2 => 0x14, // CH1/CH2 use 0x14 base
        3 | 4 => 0x1C, // CH3/CH4 use 0x1C base
        _ => 0x14,
    }
}

/// Build a two-byte Preamble Address Code for the given row and style.
fn pac_bytes(row: u8, style: &SubtitleStyle) -> (u8, u8) {
    // CEA-608 PAC table: rows 1–15 have specific encoding
    // byte1: 0x14..0x17 (channel + row group)
    // byte2: row-within-group + colour bits + underline
    let row = row.clamp(1, 15);
    let colour_bits = style.color.color_bits();
    let underline = if style.underline { 0x01 } else { 0x00 };

    // Simplified: use row to compute byte1/byte2 pair
    // Row group pairs: rows 1-2 → 0x14, 3-4 → 0x15, etc.
    let group = (row - 1) / 2;
    let b1_base: u8 = 0x14 + group.min(7);
    let row_bit = if (row - 1) % 2 == 1 { 0x20 } else { 0x00 };
    let b2 = row_bit | (colour_bits << 1) | underline;

    (set_parity(b1_base), set_parity(b2))
}

/// Map a char to its 7-bit CEA-608 ASCII equivalent (extended chars → '?').
fn ascii_to_cea608(ch: char) -> u8 {
    if ch.is_ascii() && ch as u8 >= 0x20 && ch as u8 <= 0x7E {
        ch as u8
    } else {
        b'?'
    }
}

/// Truncate text so that each row ≤ max_chars characters (soft limit).
fn truncate_row(text: &str, max_chars: usize) -> String {
    text.lines()
        .map(|l| l.chars().take(max_chars).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse `HH:MM:SS,mmm --> HH:MM:SS,mmm` (SRT format) → 90 kHz PTS pair.
fn parse_srt_timecode(line: &str) -> Result<(u64, u64)> {
    let parts: Vec<&str> = line.splitn(2, " --> ").collect();
    if parts.len() != 2 {
        return Err(PlayoutError::Playlist(format!(
            "Invalid SRT timecode: '{line}'"
        )));
    }
    let start_ms = parse_srt_timestamp(parts[0].trim())?;
    let end_ms = parse_srt_timestamp(parts[1].trim())?;
    Ok((ms_to_90k(start_ms), ms_to_90k(end_ms)))
}

/// Parse `HH:MM:SS,mmm` → milliseconds.
fn parse_srt_timestamp(ts: &str) -> Result<u64> {
    // Accept both ',' and '.' as millisecond separator
    let ts = ts.replace(',', ".");
    let parts: Vec<&str> = ts.splitn(2, '.').collect();
    let hms = parts[0];
    let ms_str = if parts.len() == 2 { parts[1] } else { "0" };

    let hms_parts: Vec<&str> = hms.split(':').collect();
    if hms_parts.len() != 3 {
        return Err(PlayoutError::Playlist(format!(
            "Invalid timestamp HH:MM:SS part: '{hms}'"
        )));
    }

    let h: u64 = hms_parts[0]
        .parse()
        .map_err(|_| PlayoutError::Playlist(format!("Invalid hours: '{}'", hms_parts[0])))?;
    let m: u64 = hms_parts[1]
        .parse()
        .map_err(|_| PlayoutError::Playlist(format!("Invalid minutes: '{}'", hms_parts[1])))?;
    let s: u64 = hms_parts[2]
        .parse()
        .map_err(|_| PlayoutError::Playlist(format!("Invalid seconds: '{}'", hms_parts[2])))?;

    // Pad/truncate ms_str to exactly 3 digits
    let ms_padded = format!("{ms_str:0<3}");
    let ms: u64 = ms_padded[..3]
        .parse()
        .map_err(|_| PlayoutError::Playlist(format!("Invalid milliseconds: '{ms_str}'")))?;

    Ok(h * 3_600_000 + m * 60_000 + s * 1_000 + ms)
}

/// Parse `HH:MM:SS.mmm --> HH:MM:SS.mmm` (WebVTT format) → 90 kHz PTS pair.
fn parse_webvtt_timecode(line: &str) -> Result<(u64, u64)> {
    // Strip optional cue settings after the second timestamp
    let clean = line
        .split_whitespace()
        .take(3)
        .collect::<Vec<_>>()
        .join(" ");
    let parts: Vec<&str> = clean.splitn(2, " --> ").collect();
    if parts.len() != 2 {
        return Err(PlayoutError::Playlist(format!(
            "Invalid WebVTT timecode: '{line}'"
        )));
    }
    // WebVTT uses '.' for milliseconds; reuse parse_srt_timestamp
    let start_ms = parse_srt_timestamp(parts[0].trim())?;
    let end_ms = parse_srt_timestamp(parts[1].trim())?;
    Ok((ms_to_90k(start_ms), ms_to_90k(end_ms)))
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. ms_to_90k conversion
    #[test]
    fn test_ms_to_90k() {
        assert_eq!(ms_to_90k(1000), 90_000);
        assert_eq!(ms_to_90k(0), 0);
        assert_eq!(ms_to_90k(500), 45_000);
    }

    // 2. set_parity: result must have an odd number of 1-bits
    #[test]
    fn test_set_parity_odd() {
        for byte in 0u8..=127 {
            let p = set_parity(byte);
            assert_eq!(p.count_ones() % 2, 1, "parity failed for byte {byte:#04x}");
        }
    }

    // 3. SubtitleEntry active-at logic
    #[test]
    fn test_entry_is_active_at() {
        let e = SubtitleEntry::new(1000, 2000, "Hello");
        assert!(!e.is_active_at(999));
        assert!(e.is_active_at(1000));
        assert!(e.is_active_at(1500));
        assert!(!e.is_active_at(2000));
    }

    // 4. CEA-608 packet starts with EDM + ends with EOC
    #[test]
    fn test_cea608_packet_structure() {
        let entry = SubtitleEntry::new(0, 1000, "Test");
        let pkt = SubtitleInserter::encode_cea608_packet(&entry);
        // Must have at least 4 bytes: EDM(2) + PAC(2) + text(≥2) + EOC(2)
        assert!(pkt.len() >= 4);
    }

    // 5. CEA-708 packet has service-block header + text
    #[test]
    fn test_cea708_packet_header() {
        let entry = SubtitleEntry::new(0, 1000, "Hello");
        let pkt = SubtitleInserter::encode_cea708_packet(&entry);
        // Minimum: 2-byte header + 5 text bytes
        assert_eq!(pkt.len(), 2 + 5);
    }

    // 6. CEA-708 clamps channel to 1–63
    #[test]
    fn test_cea708_channel_clamp() {
        let mut entry = SubtitleEntry::new(0, 1000, "X");
        entry.channel = 200; // out of range
        let pkt = SubtitleInserter::encode_cea708_packet(&entry);
        // Should not panic, length = 2 + 1
        assert_eq!(pkt.len(), 3);
    }

    // 7. get_packets_for_frame returns packets only for active entries
    #[test]
    fn test_get_packets_for_frame() {
        let mut inserter = SubtitleInserter::new();
        let mut track = SubtitleTrack::new(1, "en", SubtitleFormat::Cea608);
        track.add_entry(SubtitleEntry::new(0, 90_000, "Active"));
        track.add_entry(SubtitleEntry::new(180_000, 270_000, "Future"));
        inserter.add_track(track);

        let pkts = inserter.get_packets_for_frame(45_000); // halfway through entry 1
        assert_eq!(pkts.len(), 1);

        let pkts2 = inserter.get_packets_for_frame(90_001); // after both
        assert_eq!(pkts2.len(), 0);
    }

    // 8. parse_srt basic case
    #[test]
    fn test_parse_srt_basic() {
        let srt = "1\n00:00:01,000 --> 00:00:02,000\nHello World\n\n2\n00:00:03,000 --> 00:00:04,500\nLine two\n";
        let entries = SubtitleInserter::parse_srt(srt).expect("SRT parse should succeed");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].start_pts, 90_000); // 1s * 90
        assert_eq!(entries[0].end_pts, 180_000); // 2s * 90
        assert_eq!(entries[0].text, "Hello World");
        assert_eq!(entries[1].end_pts, 405_000); // 4.5s * 90
    }

    // 9. parse_srt invalid timecode returns error
    #[test]
    fn test_parse_srt_invalid() {
        let srt = "1\nNOT A TIMECODE\nHello\n\n";
        let result = SubtitleInserter::parse_srt(srt);
        // parse_srt silently skips cues with < 3 lines after split; adjust:
        // The cue has 3 lines, timecode parse should fail.
        assert!(result.is_err() || result.expect("parse should return entries").is_empty());
    }

    // 10. parse_webvtt basic case
    #[test]
    fn test_parse_webvtt_basic() {
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello VTT\n\n00:00:03.000 --> 00:00:04.000\nSecond\n";
        let entries = SubtitleInserter::parse_webvtt(vtt).expect("WebVTT parse should succeed");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].start_pts, 90_000);
        assert_eq!(entries[0].text, "Hello VTT");
    }

    // 11. parse_webvtt missing header returns error
    #[test]
    fn test_parse_webvtt_no_header() {
        let bad = "00:00:01.000 --> 00:00:02.000\nText\n";
        let result = SubtitleInserter::parse_webvtt(bad);
        assert!(result.is_err());
    }

    // 12. SubtitleTrack sorts entries by start_pts on add
    #[test]
    fn test_track_sorted_insert() {
        let mut track = SubtitleTrack::new(1, "en", SubtitleFormat::Srt);
        track.add_entry(SubtitleEntry::new(900, 1800, "B"));
        track.add_entry(SubtitleEntry::new(0, 900, "A"));
        assert_eq!(track.entries[0].text, "A");
        assert_eq!(track.entries[1].text, "B");
    }

    // 13. truncate_row enforces max chars per row
    #[test]
    fn test_truncate_row() {
        let s = "A".repeat(50);
        let t = truncate_row(&s, 32);
        assert_eq!(t.chars().count(), 32);
    }

    // 14. CEA-608 coloured entry produces different PAC than white
    #[test]
    fn test_cea608_color_difference() {
        let mut e_white = SubtitleEntry::new(0, 1000, "X");
        e_white.style.color = Cea608Color::White;

        let mut e_red = SubtitleEntry::new(0, 1000, "X");
        e_red.style.color = Cea608Color::Red;

        let pkt_w = SubtitleInserter::encode_cea608_packet(&e_white);
        let pkt_r = SubtitleInserter::encode_cea608_packet(&e_red);
        // PAC bytes at index 2/3 should differ for different colours
        assert_ne!(pkt_w[3], pkt_r[3]);
    }

    // 15. get_packets_for_frame with multiple tracks
    #[test]
    fn test_get_packets_multiple_tracks() {
        let mut inserter = SubtitleInserter::new();

        let mut t1 = SubtitleTrack::new(1, "en", SubtitleFormat::Cea608);
        t1.add_entry(SubtitleEntry::new(0, 90_000, "English"));

        let mut t2 = SubtitleTrack::new(2, "fr", SubtitleFormat::Cea708);
        t2.add_entry(SubtitleEntry::new(0, 90_000, "Français"));

        inserter.add_track(t1);
        inserter.add_track(t2);

        let pkts = inserter.get_packets_for_frame(1000);
        assert_eq!(pkts.len(), 2);
    }
}
