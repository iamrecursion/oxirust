//! Streaming metadata parser for incremental / partial-buffer parsing.
//!
//! When receiving media data over a network or from a pipe, the full metadata
//! block may not be available in one read.  `StreamingMetadataParser` accumulates
//! bytes and emits [`MetadataEvent`]s as soon as complete frames / fields are
//! recognized.
//!
//! # Supported formats
//!
//! - **ID3v2** — emits events after the 10-byte tag header is read and then
//!   for each complete frame.
//! - **Vorbis Comments** — emits events after the vendor string and each
//!   complete comment entry.
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::metadata_streaming::{StreamingMetadataParser, StreamingFormat};
//!
//! let mut parser = StreamingMetadataParser::new(StreamingFormat::Id3v2);
//!
//! // Feed data in chunks (simulating network reads)
//! let chunk1 = b"ID3\x04\x00\x00\x00\x00\x00\x00";
//! let events = parser.feed(chunk1);
//! // Header event emitted when 10 bytes are available
//! ```

use crate::{Error, MetadataValue};
use std::collections::HashMap;

// ---- Streaming Format ----

/// The metadata format being streamed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingFormat {
    /// ID3v2 tags (MP3).
    Id3v2,
    /// Vorbis Comments (Ogg/FLAC/Opus).
    VorbisComments,
}

// ---- Metadata Event ----

/// An event emitted by the streaming parser when a complete piece of
/// metadata has been recognized.
#[derive(Debug, Clone)]
pub enum MetadataEvent {
    /// The tag header has been parsed. Contains tag-level information.
    Header(HeaderInfo),
    /// A single metadata field has been parsed.
    Field(FieldEvent),
    /// All metadata has been parsed (end of tag).
    Complete,
    /// An error was encountered during parsing.
    Error(String),
}

/// Tag-level header information.
#[derive(Debug, Clone)]
pub struct HeaderInfo {
    /// The format of the tag.
    pub format: StreamingFormat,
    /// Total tag size in bytes (excluding the header itself).
    pub tag_size: u32,
    /// Format-specific version (e.g., 3 or 4 for ID3v2).
    pub version: u8,
}

/// A single parsed metadata field.
#[derive(Debug, Clone)]
pub struct FieldEvent {
    /// The field key (e.g., "TIT2" for ID3v2, "TITLE" for Vorbis).
    pub key: String,
    /// The field value.
    pub value: MetadataValue,
}

// ---- Parser State ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParserPhase {
    /// Waiting for enough bytes to parse the header.
    WaitingForHeader,
    /// Header parsed; reading frames/fields.
    ReadingFields,
    /// Parsing is complete.
    Done,
}

// ---- ID3v2 Streaming State ----

/// Internal state for ID3v2 incremental parsing.
#[derive(Debug, Clone)]
struct Id3v2State {
    version: u8,
    tag_size: u32,
    bytes_consumed: u32,
}

// ---- Vorbis Streaming State ----

/// Internal state for Vorbis Comment incremental parsing.
#[derive(Debug, Clone)]
struct VorbisState {
    vendor_parsed: bool,
    comment_count: u32,
    comments_read: u32,
}

// ---- Streaming Metadata Parser ----

/// Incremental metadata parser that accepts arbitrary-length byte slices
/// and emits [`MetadataEvent`]s when complete frames arrive.
#[derive(Debug)]
pub struct StreamingMetadataParser {
    format: StreamingFormat,
    phase: ParserPhase,
    buffer: Vec<u8>,
    id3v2: Option<Id3v2State>,
    vorbis: Option<VorbisState>,
    /// Collected fields so far (available via `fields()`).
    collected: HashMap<String, MetadataValue>,
}

impl StreamingMetadataParser {
    /// Create a new streaming parser for the given format.
    pub fn new(format: StreamingFormat) -> Self {
        Self {
            format,
            phase: ParserPhase::WaitingForHeader,
            buffer: Vec::new(),
            id3v2: None,
            vorbis: None,
            collected: HashMap::new(),
        }
    }

    /// Feed bytes into the parser and collect any events emitted.
    pub fn feed(&mut self, data: &[u8]) -> Vec<MetadataEvent> {
        self.buffer.extend_from_slice(data);
        let mut events = Vec::new();

        loop {
            let event = match self.phase {
                ParserPhase::WaitingForHeader => self.try_parse_header(),
                ParserPhase::ReadingFields => self.try_parse_next_field(),
                ParserPhase::Done => break,
            };

            match event {
                Some(ev) => events.push(ev),
                None => break,
            }
        }

        events
    }

    /// Whether the parser has finished (all metadata consumed).
    pub fn is_complete(&self) -> bool {
        self.phase == ParserPhase::Done
    }

    /// Access fields collected so far.
    pub fn fields(&self) -> &HashMap<String, MetadataValue> {
        &self.collected
    }

    /// Reset the parser for reuse.
    pub fn reset(&mut self) {
        self.phase = ParserPhase::WaitingForHeader;
        self.buffer.clear();
        self.id3v2 = None;
        self.vorbis = None;
        self.collected.clear();
    }

    /// The number of fields parsed so far.
    pub fn field_count(&self) -> usize {
        self.collected.len()
    }

    // ---- Header Parsing ----

    fn try_parse_header(&mut self) -> Option<MetadataEvent> {
        match self.format {
            StreamingFormat::Id3v2 => self.try_parse_id3v2_header(),
            StreamingFormat::VorbisComments => self.try_parse_vorbis_header(),
        }
    }

    fn try_parse_id3v2_header(&mut self) -> Option<MetadataEvent> {
        // Need at least 10 bytes for ID3v2 header
        if self.buffer.len() < 10 {
            return None;
        }

        // Check magic
        if &self.buffer[0..3] != b"ID3" {
            self.phase = ParserPhase::Done;
            return Some(MetadataEvent::Error("Not an ID3v2 tag".to_string()));
        }

        let version = self.buffer[3];
        if version != 3 && version != 4 {
            self.phase = ParserPhase::Done;
            return Some(MetadataEvent::Error(format!(
                "Unsupported ID3v2 version: {version}"
            )));
        }

        // Synchsafe tag size
        let tag_size = (u32::from(self.buffer[6] & 0x7F) << 21)
            | (u32::from(self.buffer[7] & 0x7F) << 14)
            | (u32::from(self.buffer[8] & 0x7F) << 7)
            | u32::from(self.buffer[9] & 0x7F);

        self.id3v2 = Some(Id3v2State {
            version,
            tag_size,
            bytes_consumed: 0,
        });

        // Remove header bytes from buffer
        self.buffer.drain(..10);
        self.phase = ParserPhase::ReadingFields;

        Some(MetadataEvent::Header(HeaderInfo {
            format: StreamingFormat::Id3v2,
            tag_size,
            version,
        }))
    }

    fn try_parse_vorbis_header(&mut self) -> Option<MetadataEvent> {
        // Need at least 4 bytes for vendor string length
        if self.buffer.len() < 4 {
            return None;
        }

        let vendor_len = u32::from_le_bytes([
            self.buffer[0],
            self.buffer[1],
            self.buffer[2],
            self.buffer[3],
        ]) as usize;

        // Need vendor_len + 4 (comment_count)
        let needed = 4 + vendor_len + 4;
        if self.buffer.len() < needed {
            return None;
        }

        let vendor = String::from_utf8_lossy(&self.buffer[4..4 + vendor_len]).to_string();
        let comment_count = u32::from_le_bytes([
            self.buffer[4 + vendor_len],
            self.buffer[4 + vendor_len + 1],
            self.buffer[4 + vendor_len + 2],
            self.buffer[4 + vendor_len + 3],
        ]);

        self.collected
            .insert("VENDOR".to_string(), MetadataValue::Text(vendor));

        self.vorbis = Some(VorbisState {
            vendor_parsed: true,
            comment_count,
            comments_read: 0,
        });

        self.buffer.drain(..needed);
        self.phase = if comment_count == 0 {
            ParserPhase::Done
        } else {
            ParserPhase::ReadingFields
        };

        let event = MetadataEvent::Header(HeaderInfo {
            format: StreamingFormat::VorbisComments,
            tag_size: 0, // not applicable for Vorbis
            version: 0,
        });

        if comment_count == 0 {
            // We'll return Header; Complete will come on next feed()
            // Actually, let the loop pick up Done state
        }

        Some(event)
    }

    // ---- Field Parsing ----

    fn try_parse_next_field(&mut self) -> Option<MetadataEvent> {
        match self.format {
            StreamingFormat::Id3v2 => self.try_parse_id3v2_frame(),
            StreamingFormat::VorbisComments => self.try_parse_vorbis_field(),
        }
    }

    fn try_parse_id3v2_frame(&mut self) -> Option<MetadataEvent> {
        let state = self.id3v2.as_mut()?;

        // Check if we've consumed all frame data
        if state.bytes_consumed >= state.tag_size {
            self.phase = ParserPhase::Done;
            return Some(MetadataEvent::Complete);
        }

        // Need at least 10 bytes for frame header
        if self.buffer.len() < 10 {
            return None;
        }

        // Check for padding (null bytes indicate end of frames)
        if self.buffer[0] == 0 {
            self.phase = ParserPhase::Done;
            return Some(MetadataEvent::Complete);
        }

        // Read frame ID (4 bytes)
        let frame_id = match std::str::from_utf8(&self.buffer[0..4]) {
            Ok(s) => s.to_string(),
            Err(_) => {
                self.phase = ParserPhase::Done;
                return Some(MetadataEvent::Error(
                    "Invalid frame ID encoding".to_string(),
                ));
            }
        };

        // Read frame size
        let frame_size = if state.version == 4 {
            // Synchsafe
            (u32::from(self.buffer[4] & 0x7F) << 21)
                | (u32::from(self.buffer[5] & 0x7F) << 14)
                | (u32::from(self.buffer[6] & 0x7F) << 7)
                | u32::from(self.buffer[7] & 0x7F)
        } else {
            u32::from_be_bytes([
                self.buffer[4],
                self.buffer[5],
                self.buffer[6],
                self.buffer[7],
            ])
        } as usize;

        // Need header (10) + frame data
        if self.buffer.len() < 10 + frame_size {
            return None;
        }

        let frame_data = self.buffer[10..10 + frame_size].to_vec();
        let total = 10 + frame_size;
        self.buffer.drain(..total);
        state.bytes_consumed += total as u32;

        // Parse text frames
        let value = if frame_id.starts_with('T') && !frame_data.is_empty() {
            let encoding = frame_data[0];
            let text_bytes = &frame_data[1..];
            let text = match encoding {
                0 => String::from_utf8_lossy(text_bytes).to_string(),
                3 => String::from_utf8_lossy(text_bytes).to_string(),
                1 | 2 => {
                    // UTF-16 - simplified
                    String::from_utf8_lossy(text_bytes).to_string()
                }
                _ => String::from_utf8_lossy(text_bytes).to_string(),
            };
            MetadataValue::Text(text.trim_end_matches('\0').to_string())
        } else {
            MetadataValue::Binary(frame_data)
        };

        self.collected.insert(frame_id.clone(), value.clone());

        Some(MetadataEvent::Field(FieldEvent {
            key: frame_id,
            value,
        }))
    }

    fn try_parse_vorbis_field(&mut self) -> Option<MetadataEvent> {
        let state = self.vorbis.as_mut()?;

        if !state.vendor_parsed {
            return None;
        }

        if state.comments_read >= state.comment_count {
            self.phase = ParserPhase::Done;
            return Some(MetadataEvent::Complete);
        }

        // Need 4 bytes for comment length
        if self.buffer.len() < 4 {
            return None;
        }

        let comment_len = u32::from_le_bytes([
            self.buffer[0],
            self.buffer[1],
            self.buffer[2],
            self.buffer[3],
        ]) as usize;

        if self.buffer.len() < 4 + comment_len {
            return None;
        }

        let comment = String::from_utf8_lossy(&self.buffer[4..4 + comment_len]).to_string();
        self.buffer.drain(..4 + comment_len);
        state.comments_read += 1;

        // Parse NAME=value
        let (key, value) = if let Some(eq_pos) = comment.find('=') {
            let name = comment[..eq_pos].to_uppercase();
            let val = comment[eq_pos + 1..].to_string();
            (name, MetadataValue::Text(val))
        } else {
            (comment, MetadataValue::Text(String::new()))
        };

        // Handle multi-value
        if let Some(existing) = self.collected.get(&key) {
            match existing {
                MetadataValue::Text(t) => {
                    if let MetadataValue::Text(ref new_val) = value {
                        let list = vec![t.clone(), new_val.clone()];
                        self.collected
                            .insert(key.clone(), MetadataValue::TextList(list));
                    }
                }
                MetadataValue::TextList(list) => {
                    if let MetadataValue::Text(ref new_val) = value {
                        let mut new_list = list.clone();
                        new_list.push(new_val.clone());
                        self.collected
                            .insert(key.clone(), MetadataValue::TextList(new_list));
                    }
                }
                _ => {
                    self.collected.insert(key.clone(), value.clone());
                }
            }
        } else {
            self.collected.insert(key.clone(), value.clone());
        }

        // Check if done
        if state.comments_read >= state.comment_count {
            self.phase = ParserPhase::Done;
        }

        Some(MetadataEvent::Field(FieldEvent { key, value }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a minimal ID3v2.4 tag
    fn build_id3v2_tag(fields: &[(&str, &str)]) -> Vec<u8> {
        let mut frames = Vec::new();
        for (id, text) in fields {
            let mut frame_body = vec![3u8]; // UTF-8
            frame_body.extend_from_slice(text.as_bytes());

            let frame_size = frame_body.len() as u32;
            frames.extend_from_slice(id.as_bytes());
            // synchsafe size
            frames.push(((frame_size >> 21) & 0x7F) as u8);
            frames.push(((frame_size >> 14) & 0x7F) as u8);
            frames.push(((frame_size >> 7) & 0x7F) as u8);
            frames.push((frame_size & 0x7F) as u8);
            frames.extend_from_slice(&[0u8; 2]); // flags
            frames.extend_from_slice(&frame_body);
        }

        let tag_size = frames.len() as u32;
        let mut data = Vec::new();
        data.extend_from_slice(b"ID3");
        data.push(4); // v2.4
        data.push(0); // revision
        data.push(0); // flags
        data.push(((tag_size >> 21) & 0x7F) as u8);
        data.push(((tag_size >> 14) & 0x7F) as u8);
        data.push(((tag_size >> 7) & 0x7F) as u8);
        data.push((tag_size & 0x7F) as u8);
        data.extend_from_slice(&frames);
        data
    }

    // Helper: build Vorbis Comment data
    fn build_vorbis_data(vendor: &str, comments: &[(&str, &str)]) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        data.extend_from_slice(vendor.as_bytes());
        data.extend_from_slice(&(comments.len() as u32).to_le_bytes());
        for (key, value) in comments {
            let comment = format!("{key}={value}");
            data.extend_from_slice(&(comment.len() as u32).to_le_bytes());
            data.extend_from_slice(comment.as_bytes());
        }
        data
    }

    #[test]
    fn test_streaming_parser_new() {
        let parser = StreamingMetadataParser::new(StreamingFormat::Id3v2);
        assert!(!parser.is_complete());
        assert_eq!(parser.field_count(), 0);
    }

    #[test]
    fn test_streaming_id3v2_full_feed() {
        let tag = build_id3v2_tag(&[("TIT2", "My Song"), ("TPE1", "Artist")]);
        let mut parser = StreamingMetadataParser::new(StreamingFormat::Id3v2);

        let events = parser.feed(&tag);

        // Should get: Header, Field(TIT2), Field(TPE1), Complete
        let header_count = events
            .iter()
            .filter(|e| matches!(e, MetadataEvent::Header(_)))
            .count();
        let field_count = events
            .iter()
            .filter(|e| matches!(e, MetadataEvent::Field(_)))
            .count();
        let complete_count = events
            .iter()
            .filter(|e| matches!(e, MetadataEvent::Complete))
            .count();

        assert_eq!(header_count, 1);
        assert_eq!(field_count, 2);
        assert_eq!(complete_count, 1);
        assert!(parser.is_complete());
        assert_eq!(parser.field_count(), 2);

        // Check collected values
        let title = parser.fields().get("TIT2");
        assert!(title.is_some());
        assert_eq!(title.and_then(|v| v.as_text()), Some("My Song"));
    }

    #[test]
    fn test_streaming_id3v2_chunked_feed() {
        let tag = build_id3v2_tag(&[("TIT2", "Title"), ("TALB", "Album")]);
        let mut parser = StreamingMetadataParser::new(StreamingFormat::Id3v2);

        // Feed header only (first 10 bytes)
        let events1 = parser.feed(&tag[..10]);
        assert_eq!(events1.len(), 1);
        assert!(matches!(&events1[0], MetadataEvent::Header(_)));
        assert!(!parser.is_complete());

        // Feed remaining data
        let events2 = parser.feed(&tag[10..]);
        let field_count = events2
            .iter()
            .filter(|e| matches!(e, MetadataEvent::Field(_)))
            .count();
        assert_eq!(field_count, 2);
        assert!(parser.is_complete());
    }

    #[test]
    fn test_streaming_id3v2_byte_by_byte() {
        let tag = build_id3v2_tag(&[("TIT2", "Hi")]);
        let mut parser = StreamingMetadataParser::new(StreamingFormat::Id3v2);

        let mut all_events = Vec::new();
        for byte in &tag {
            let events = parser.feed(&[*byte]);
            all_events.extend(events);
        }

        assert!(parser.is_complete());
        let field_events: Vec<_> = all_events
            .iter()
            .filter(|e| matches!(e, MetadataEvent::Field(_)))
            .collect();
        assert_eq!(field_events.len(), 1);
    }

    #[test]
    fn test_streaming_id3v2_invalid_magic() {
        let mut parser = StreamingMetadataParser::new(StreamingFormat::Id3v2);
        let events = parser.feed(b"NOT_ID3_AT_ALL");
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], MetadataEvent::Error(_)));
        assert!(parser.is_complete());
    }

    #[test]
    fn test_streaming_vorbis_full_feed() {
        let data = build_vorbis_data("OxiMedia", &[("TITLE", "Song"), ("ARTIST", "Alice")]);
        let mut parser = StreamingMetadataParser::new(StreamingFormat::VorbisComments);

        let events = parser.feed(&data);

        let header_count = events
            .iter()
            .filter(|e| matches!(e, MetadataEvent::Header(_)))
            .count();
        let field_count = events
            .iter()
            .filter(|e| matches!(e, MetadataEvent::Field(_)))
            .count();

        assert_eq!(header_count, 1);
        assert_eq!(field_count, 2);
        assert!(parser.is_complete());
        assert_eq!(
            parser.fields().get("TITLE").and_then(|v| v.as_text()),
            Some("Song")
        );
        assert_eq!(
            parser.fields().get("ARTIST").and_then(|v| v.as_text()),
            Some("Alice")
        );
    }

    #[test]
    fn test_streaming_vorbis_chunked_feed() {
        let data = build_vorbis_data("TestEncoder", &[("ALBUM", "My Album"), ("DATE", "2025")]);
        let mut parser = StreamingMetadataParser::new(StreamingFormat::VorbisComments);

        // Feed just the first few bytes (not enough for header)
        let events1 = parser.feed(&data[..2]);
        assert!(events1.is_empty());

        // Feed more, completing the header + some fields
        let events2 = parser.feed(&data[2..]);
        assert!(!events2.is_empty());
        assert!(parser.is_complete());
    }

    #[test]
    fn test_streaming_vorbis_multivalue() {
        let data = build_vorbis_data("Enc", &[("ARTIST", "Alice"), ("ARTIST", "Bob")]);
        let mut parser = StreamingMetadataParser::new(StreamingFormat::VorbisComments);

        let _events = parser.feed(&data);
        assert!(parser.is_complete());

        let artists = parser.fields().get("ARTIST");
        assert!(artists.is_some());
        match artists {
            Some(MetadataValue::TextList(list)) => {
                assert_eq!(list.len(), 2);
                assert_eq!(list[0], "Alice");
                assert_eq!(list[1], "Bob");
            }
            _ => panic!("Expected TextList for multi-value ARTIST"),
        }
    }

    #[test]
    fn test_streaming_vorbis_empty() {
        let data = build_vorbis_data("Empty", &[]);
        let mut parser = StreamingMetadataParser::new(StreamingFormat::VorbisComments);

        let events = parser.feed(&data);
        let header_count = events
            .iter()
            .filter(|e| matches!(e, MetadataEvent::Header(_)))
            .count();
        assert_eq!(header_count, 1);
        // With 0 comments, phase transitions to Done immediately
        // No field events expected
    }

    #[test]
    fn test_streaming_parser_reset() {
        let tag = build_id3v2_tag(&[("TIT2", "First")]);
        let mut parser = StreamingMetadataParser::new(StreamingFormat::Id3v2);

        parser.feed(&tag);
        assert!(parser.is_complete());
        assert_eq!(parser.field_count(), 1);

        parser.reset();
        assert!(!parser.is_complete());
        assert_eq!(parser.field_count(), 0);

        // Re-feed a different tag
        let tag2 = build_id3v2_tag(&[("TALB", "Album")]);
        parser.feed(&tag2);
        assert!(parser.is_complete());
        assert_eq!(parser.field_count(), 1);
        assert_eq!(
            parser.fields().get("TALB").and_then(|v| v.as_text()),
            Some("Album")
        );
    }

    #[test]
    fn test_header_info_fields() {
        let tag = build_id3v2_tag(&[("TIT2", "T")]);
        let mut parser = StreamingMetadataParser::new(StreamingFormat::Id3v2);
        let events = parser.feed(&tag);

        if let MetadataEvent::Header(info) = &events[0] {
            assert_eq!(info.format, StreamingFormat::Id3v2);
            assert_eq!(info.version, 4);
            assert!(info.tag_size > 0);
        } else {
            panic!("Expected Header event");
        }
    }

    #[test]
    fn test_field_event_contents() {
        let tag = build_id3v2_tag(&[("TIT2", "TestSong")]);
        let mut parser = StreamingMetadataParser::new(StreamingFormat::Id3v2);
        let events = parser.feed(&tag);

        let field_events: Vec<_> = events
            .into_iter()
            .filter_map(|e| {
                if let MetadataEvent::Field(f) = e {
                    Some(f)
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(field_events.len(), 1);
        assert_eq!(field_events[0].key, "TIT2");
        assert_eq!(field_events[0].value.as_text(), Some("TestSong"));
    }

    #[test]
    fn test_streaming_vorbis_vendor_stored() {
        let data = build_vorbis_data("MyEncoder 1.0", &[("TITLE", "T")]);
        let mut parser = StreamingMetadataParser::new(StreamingFormat::VorbisComments);
        parser.feed(&data);

        assert_eq!(
            parser.fields().get("VENDOR").and_then(|v| v.as_text()),
            Some("MyEncoder 1.0")
        );
    }
}
