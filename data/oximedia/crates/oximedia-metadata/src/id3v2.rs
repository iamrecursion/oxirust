//! ID3v2 tag parsing and writing support.
//!
//! Supports ID3v2.3 and ID3v2.4 tags with all standard frame types.
//!
//! # Frame Format
//!
//! ID3v2 tags consist of a header followed by frames. Each frame has:
//! - Frame ID (4 bytes)
//! - Size (4 bytes)
//! - Flags (2 bytes)
//! - Frame data
//!
//! # Common Frames
//!
//! - **TIT2**: Title
//! - **TPE1**: Artist
//! - **TALB**: Album
//! - **TPE2**: Album Artist
//! - **TYER**: Year
//! - **TCON**: Genre
//! - **TRCK**: Track number
//! - **APIC**: Attached picture

use crate::{Error, Metadata, MetadataFormat, MetadataValue, Picture, PictureType};
use encoding_rs::{CoderResult, Encoding, UTF_16BE, UTF_16LE, UTF_8, WINDOWS_1252};

/// Decode `bytes` with a single reused `Decoder` and no intermediate `Cow`,
/// preserving the exact BOM semantics of [`Encoding::decode`].
///
/// `Encoding`'s convenience [`Encoding::decode`] both (a) constructs a fresh
/// `Decoder` and (b) returns a `Cow<str>` that, for any non-borrowable input,
/// heap-allocates. For ID3v2 text frames — which are decoded per frame — that
/// is wasteful. This helper instead pre-sizes a single `String` to the exact
/// worst-case UTF-8 length and decodes directly into it in one shot.
///
/// # BOM handling (identical to `Encoding::decode`)
///
/// Like the convenience method, this first calls [`Encoding::for_bom`]: if the
/// input starts with a UTF-8 / UTF-16LE / UTF-16BE BOM, that encoding is
/// selected and the BOM stripped; otherwise the supplied `enc` is used. The
/// body is then decoded with BOM handling already resolved, so the per-frame
/// reused decoder is a faithful, allocation-light replacement — for *every*
/// `TextEncoding` case, including BOM-bearing UTF-16 frames.
fn decode_static(enc: &'static Encoding, bytes: &[u8]) -> String {
    // Mirror `Encoding::decode`: a leading BOM overrides `enc` and is stripped.
    let (encoding, without_bom) = match Encoding::for_bom(bytes) {
        Some((bom_encoding, bom_length)) => (bom_encoding, &bytes[bom_length..]),
        None => (enc, bytes),
    };

    let mut decoder = encoding.new_decoder_without_bom_handling();
    // `max_utf8_buffer_length` returns `None` only on `usize` overflow, which
    // cannot happen for an in-memory `&[u8]` slice (its length already fits in
    // `usize`). Fall back to the convenience decode if it ever did, so we never
    // panic and never under-allocate.
    let capacity = match decoder.max_utf8_buffer_length(without_bom.len()) {
        Some(cap) => cap,
        None => {
            let (decoded, _had_errors) = encoding.decode_without_bom_handling(without_bom);
            return decoded.into_owned();
        }
    };
    let mut out = String::with_capacity(capacity);
    // With a worst-case-sized buffer and `last = true`, a single call drains the
    // whole input (`CoderResult::InputEmpty`). The buffer is sized so no output
    // overflow (`OutputFull`) is possible; if the impl ever reported otherwise,
    // fall back rather than silently truncate.
    let (result, _read, _replaced) = decoder.decode_to_string(without_bom, &mut out, true);
    match result {
        CoderResult::InputEmpty => out,
        CoderResult::OutputFull => {
            let (decoded, _had_errors) = encoding.decode_without_bom_handling(without_bom);
            decoded.into_owned()
        }
    }
}

/// ID3v2 header size (10 bytes)
const ID3V2_HEADER_SIZE: usize = 10;

/// ID3v2 frame header size (10 bytes)
const ID3V2_FRAME_HEADER_SIZE: usize = 10;

/// ID3v2 version 2.3
const ID3V2_VERSION_3: u8 = 3;

/// ID3v2 version 2.4
const ID3V2_VERSION_4: u8 = 4;

/// Default text encoding for writing.
///
/// ID3v2.4 introduced UTF-8 (encoding byte 3) as a valid text encoding.
/// We prefer UTF-8 for v2.4 tags because it is compact for ASCII/Latin text,
/// fully Unicode-capable, and widely supported by modern software.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Id3v2WriteEncoding {
    /// Prefer UTF-8 for text frames (ID3v2.4 only, recommended).
    Utf8,
    /// Prefer UTF-16 with BOM for text frames (compatible with v2.3 and v2.4).
    Utf16,
    /// Prefer ISO-8859-1 / Latin-1 (only safe for ASCII-range text).
    Latin1,
}

impl Default for Id3v2WriteEncoding {
    fn default() -> Self {
        Self::Utf8
    }
}

impl Id3v2WriteEncoding {
    /// Return the corresponding `TextEncoding`.
    fn to_text_encoding(self) -> TextEncoding {
        match self {
            Self::Utf8 => TextEncoding::Utf8,
            Self::Utf16 => TextEncoding::Utf16,
            Self::Latin1 => TextEncoding::Latin1,
        }
    }
}

/// Options for writing ID3v2 tags.
#[derive(Debug, Clone)]
pub struct Id3v2WriteOptions {
    /// ID3v2 version to write (3 or 4). Default: 4.
    pub version: u8,
    /// Text encoding preference. Default: UTF-8 (v2.4).
    pub encoding: Id3v2WriteEncoding,
}

impl Default for Id3v2WriteOptions {
    fn default() -> Self {
        Self {
            version: ID3V2_VERSION_4,
            encoding: Id3v2WriteEncoding::Utf8,
        }
    }
}

/// Text encoding types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextEncoding {
    /// ISO-8859-1 (Latin-1)
    Latin1,
    /// UTF-16 with BOM
    Utf16,
    /// UTF-16BE without BOM
    Utf16Be,
    /// UTF-8
    Utf8,
}

impl TextEncoding {
    /// Create from encoding byte.
    fn from_byte(byte: u8) -> Self {
        match byte {
            0 => Self::Latin1,
            1 => Self::Utf16,
            2 => Self::Utf16Be,
            3 => Self::Utf8,
            _ => Self::Latin1,
        }
    }

    /// Convert to encoding byte.
    fn to_byte(self) -> u8 {
        match self {
            Self::Latin1 => 0,
            Self::Utf16 => 1,
            Self::Utf16Be => 2,
            Self::Utf8 => 3,
        }
    }

    /// Decode bytes to string.
    fn decode(self, bytes: &[u8]) -> Result<String, Error> {
        match self {
            // Latin-1 / UTF-8 / explicit UTF-16BE are all fixed encodings, so the
            // decoder-reuse path applies directly (no BOM to interpret).
            Self::Latin1 => Ok(decode_static(WINDOWS_1252, bytes)),
            Self::Utf8 => Ok(decode_static(UTF_8, bytes)),
            Self::Utf16Be => Ok(decode_static(UTF_16BE, bytes)),
            Self::Utf16 => {
                // Inspect the BOM to choose endianness, then decode the *remaining*
                // (BOM-stripped) bytes with the now-fixed encoding. This preserves
                // the original BOM semantics exactly: the BOM selects LE vs BE and
                // is then skipped, so it is safe to use the without-BOM-handling
                // decoder for the body.
                if bytes.len() >= 2 {
                    let (encoding, start_pos) = if bytes[0] == 0xFF && bytes[1] == 0xFE {
                        (UTF_16LE, 2)
                    } else if bytes[0] == 0xFE && bytes[1] == 0xFF {
                        (UTF_16BE, 2)
                    } else {
                        (UTF_16LE, 0)
                    };
                    Ok(decode_static(encoding, &bytes[start_pos..]))
                } else {
                    Ok(String::new())
                }
            }
        }
    }

    /// Encode string to bytes.
    #[allow(dead_code)]
    fn encode(self, text: &str) -> Vec<u8> {
        match self {
            Self::Latin1 => {
                let (encoded, _, _) = WINDOWS_1252.encode(text);
                encoded.into_owned()
            }
            Self::Utf16 => {
                // UTF-16LE with BOM -- manually encode because encoding_rs
                // does not support UTF-16 *encoding* (only decoding).
                let mut result = vec![0xFF, 0xFE]; // LE BOM
                for code_unit in text.encode_utf16() {
                    result.extend_from_slice(&code_unit.to_le_bytes());
                }
                result
            }
            Self::Utf16Be => {
                // Manually encode to UTF-16BE.
                let mut result = Vec::new();
                for code_unit in text.encode_utf16() {
                    result.extend_from_slice(&code_unit.to_be_bytes());
                }
                result
            }
            Self::Utf8 => text.as_bytes().to_vec(),
        }
    }
}

/// Parse ID3v2 tags from data.
///
/// # Errors
///
/// Returns an error if the data is not a valid ID3v2 tag.
#[allow(clippy::too_many_lines)]
pub fn parse(data: &[u8]) -> Result<Metadata, Error> {
    if data.len() < ID3V2_HEADER_SIZE {
        return Err(Error::ParseError(
            "Data too short for ID3v2 header".to_string(),
        ));
    }

    // Check ID3 identifier
    if &data[0..3] != b"ID3" {
        return Err(Error::ParseError("Not an ID3v2 tag".to_string()));
    }

    let version = data[3];
    let _revision = data[4];
    let flags = data[5];

    // Only support v2.3 and v2.4
    if version != ID3V2_VERSION_3 && version != ID3V2_VERSION_4 {
        return Err(Error::Unsupported(format!("ID3v2.{version} not supported")));
    }

    // Parse tag size (synchsafe integer)
    let tag_size = decode_synchsafe_int(&data[6..10])?;

    // Check for extended header
    let mut frame_data_start = ID3V2_HEADER_SIZE;
    if flags & 0x40 != 0 {
        // Extended header present
        if data.len() < frame_data_start + 4 {
            return Err(Error::ParseError("Extended header too short".to_string()));
        }
        let ext_header_size = if version == ID3V2_VERSION_4 {
            decode_synchsafe_int(&data[frame_data_start..frame_data_start + 4])?
        } else {
            u32::from_be_bytes([
                data[frame_data_start],
                data[frame_data_start + 1],
                data[frame_data_start + 2],
                data[frame_data_start + 3],
            ])
        };
        frame_data_start += ext_header_size as usize;
    }

    let tag_end = ID3V2_HEADER_SIZE + tag_size as usize;
    if data.len() < tag_end {
        return Err(Error::ParseError(
            "Tag size exceeds data length".to_string(),
        ));
    }

    let mut metadata = Metadata::new(MetadataFormat::Id3v2);
    let mut pos = frame_data_start;

    // Parse frames
    while pos + ID3V2_FRAME_HEADER_SIZE <= tag_end {
        // Check for padding (null bytes)
        if data[pos] == 0 {
            break;
        }

        // Parse frame header
        let frame_id = std::str::from_utf8(&data[pos..pos + 4])
            .map_err(|e| Error::ParseError(format!("Invalid frame ID: {e}")))?
            .to_string();

        let frame_size = if version == ID3V2_VERSION_4 {
            decode_synchsafe_int(&data[pos + 4..pos + 8])?
        } else {
            u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
        } as usize;

        let _frame_flags = u16::from_be_bytes([data[pos + 8], data[pos + 9]]);

        pos += ID3V2_FRAME_HEADER_SIZE;

        if pos + frame_size > tag_end {
            return Err(Error::ParseError("Frame size exceeds tag size".to_string()));
        }

        let frame_data = &data[pos..pos + frame_size];
        pos += frame_size;

        // Parse frame data based on frame ID
        let value = parse_frame(&frame_id, frame_data)?;
        metadata.insert(frame_id, value);
    }

    Ok(metadata)
}

/// Parse a single frame.
fn parse_frame(frame_id: &str, data: &[u8]) -> Result<MetadataValue, Error> {
    if data.is_empty() {
        return Ok(MetadataValue::Text(String::new()));
    }

    // Text frames (T***)
    if frame_id.starts_with('T') && frame_id != "TXXX" {
        return parse_text_frame(data);
    }

    // URL frames (W***)
    if frame_id.starts_with('W') && frame_id != "WXXX" {
        return parse_url_frame(data);
    }

    // Special frames
    match frame_id {
        "COMM" => parse_comment_frame(data),
        "USLT" => parse_lyrics_frame(data),
        "APIC" => parse_picture_frame(data),
        "TXXX" | "WXXX" => parse_text_frame(data),
        _ => Ok(MetadataValue::Binary(data.to_vec())),
    }
}

/// Parse text frame (T***).
fn parse_text_frame(data: &[u8]) -> Result<MetadataValue, Error> {
    if data.is_empty() {
        return Ok(MetadataValue::Text(String::new()));
    }

    let encoding = TextEncoding::from_byte(data[0]);
    let text = encoding.decode(&data[1..])?;

    // Remove null terminator if present
    let text = text.trim_end_matches('\0');

    Ok(MetadataValue::Text(text.to_string()))
}

/// Parse URL frame (W***).
fn parse_url_frame(data: &[u8]) -> Result<MetadataValue, Error> {
    // URLs are always Latin-1 encoded
    let text = TextEncoding::Latin1.decode(data)?;
    Ok(MetadataValue::Text(text.trim_end_matches('\0').to_string()))
}

/// Parse comment frame (COMM).
fn parse_comment_frame(data: &[u8]) -> Result<MetadataValue, Error> {
    if data.len() < 4 {
        return Ok(MetadataValue::Text(String::new()));
    }

    let encoding = TextEncoding::from_byte(data[0]);
    // Skip language (3 bytes) and short description
    let mut pos = 4;

    // Find null terminator for description
    while pos < data.len() && data[pos] != 0 {
        pos += 1;
    }
    pos += 1; // Skip null terminator

    if pos >= data.len() {
        return Ok(MetadataValue::Text(String::new()));
    }

    let text = encoding.decode(&data[pos..])?;
    Ok(MetadataValue::Text(text.trim_end_matches('\0').to_string()))
}

/// Parse lyrics frame (USLT).
fn parse_lyrics_frame(data: &[u8]) -> Result<MetadataValue, Error> {
    // Same format as COMM
    parse_comment_frame(data)
}

/// Parse picture frame (APIC).
fn parse_picture_frame(data: &[u8]) -> Result<MetadataValue, Error> {
    if data.is_empty() {
        return Err(Error::ParseError("Empty picture frame".to_string()));
    }

    let encoding = TextEncoding::from_byte(data[0]);
    let mut pos = 1;

    // Read MIME type (null-terminated Latin-1 string)
    let mime_start = pos;
    while pos < data.len() && data[pos] != 0 {
        pos += 1;
    }
    let mime_type = String::from_utf8_lossy(&data[mime_start..pos]).to_string();
    pos += 1; // Skip null terminator

    if pos >= data.len() {
        return Err(Error::ParseError("Picture frame truncated".to_string()));
    }

    // Read picture type
    let picture_type = PictureType::from_id3v2_code(data[pos]);
    pos += 1;

    // Read description (null-terminated string)
    let desc_start = pos;
    while pos < data.len() && data[pos] != 0 {
        pos += 1;
    }
    let description = encoding.decode(&data[desc_start..pos])?;
    pos += 1; // Skip null terminator

    // Remaining data is picture data
    let picture_data = data[pos..].to_vec();

    let picture = Picture::new(mime_type, picture_type, picture_data).with_description(description);

    Ok(MetadataValue::Picture(picture))
}

/// Write ID3v2 tags to bytes using default options (v2.4 with UTF-8).
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn write(metadata: &Metadata) -> Result<Vec<u8>, Error> {
    write_with_options(metadata, &Id3v2WriteOptions::default())
}

/// Write ID3v2 tags to bytes with the given options.
///
/// When `options.version` is 4, the writer uses synchsafe frame sizes
/// and honours the encoding preference (UTF-8 by default).  When
/// `options.version` is 3, frames use plain big-endian sizes and the
/// encoding falls back to UTF-16 (since v2.3 does not support UTF-8).
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn write_with_options(
    metadata: &Metadata,
    options: &Id3v2WriteOptions,
) -> Result<Vec<u8>, Error> {
    let version = options.version;
    // v2.3 does not support UTF-8; fall back to UTF-16.
    let encoding = if version == ID3V2_VERSION_3 && options.encoding == Id3v2WriteEncoding::Utf8 {
        Id3v2WriteEncoding::Utf16
    } else {
        options.encoding
    };
    let text_enc = encoding.to_text_encoding();

    let mut result = Vec::new();

    // Write header
    result.extend_from_slice(b"ID3");
    result.push(version);
    result.push(0); // Revision
    result.push(0); // Flags

    // Collect frames
    let mut frames = Vec::new();
    for (key, value) in metadata.fields() {
        let frame_data = write_frame(key, value, version, text_enc)?;
        frames.extend_from_slice(&frame_data);
    }

    // Write size (synchsafe integer -- always synchsafe in the tag header)
    let size = encode_synchsafe_int(frames.len() as u32);
    result.extend_from_slice(&size);

    // Write frames
    result.extend_from_slice(&frames);

    Ok(result)
}

/// Write a single frame.
fn write_frame(
    frame_id: &str,
    value: &MetadataValue,
    version: u8,
    text_enc: TextEncoding,
) -> Result<Vec<u8>, Error> {
    let mut result = Vec::new();

    // Write frame ID
    if frame_id.len() != 4 {
        return Err(Error::WriteError(format!(
            "Invalid frame ID length: {frame_id}"
        )));
    }
    result.extend_from_slice(frame_id.as_bytes());

    // Prepare frame data
    let frame_data = match value {
        MetadataValue::Text(text) => write_text_frame(text, text_enc),
        MetadataValue::Integer(i) => write_text_frame(&i.to_string(), text_enc),
        MetadataValue::Picture(pic) => write_picture_frame(pic, text_enc),
        MetadataValue::Binary(data) => data.clone(),
        _ => {
            return Err(Error::WriteError(
                "Unsupported value type for ID3v2".to_string(),
            ))
        }
    };

    // Write size (v2.4 = synchsafe, v2.3 = plain big-endian)
    let size_u32 = frame_data.len() as u32;
    if version == ID3V2_VERSION_4 {
        result.extend_from_slice(&encode_synchsafe_int(size_u32));
    } else {
        result.extend_from_slice(&size_u32.to_be_bytes());
    }

    // Write flags
    result.extend_from_slice(&[0, 0]);

    // Write data
    result.extend_from_slice(&frame_data);

    Ok(result)
}

/// Write text frame data with the specified encoding.
fn write_text_frame(text: &str, enc: TextEncoding) -> Vec<u8> {
    let mut result = Vec::new();
    result.push(enc.to_byte());
    result.extend_from_slice(&enc.encode(text));
    result
}

/// Write picture frame data with the specified encoding.
fn write_picture_frame(picture: &Picture, enc: TextEncoding) -> Vec<u8> {
    let mut result = Vec::new();

    // Encoding byte
    result.push(enc.to_byte());

    // MIME type (always Latin-1, null-terminated)
    result.extend_from_slice(picture.mime_type.as_bytes());
    result.push(0);

    // Picture type
    result.push(picture.picture_type.to_id3v2_code());

    // Description (in selected encoding, null-terminated)
    result.extend_from_slice(&enc.encode(&picture.description));
    result.push(0);

    // Picture data
    result.extend_from_slice(&picture.data);

    result
}

/// Decode synchsafe integer.
fn decode_synchsafe_int(bytes: &[u8]) -> Result<u32, Error> {
    if bytes.len() < 4 {
        return Err(Error::ParseError(
            "Not enough bytes for synchsafe int".to_string(),
        ));
    }

    Ok(u32::from(bytes[0] & 0x7F) << 21
        | u32::from(bytes[1] & 0x7F) << 14
        | u32::from(bytes[2] & 0x7F) << 7
        | u32::from(bytes[3] & 0x7F))
}

/// Encode synchsafe integer.
fn encode_synchsafe_int(value: u32) -> [u8; 4] {
    [
        ((value >> 21) & 0x7F) as u8,
        ((value >> 14) & 0x7F) as u8,
        ((value >> 7) & 0x7F) as u8,
        (value & 0x7F) as u8,
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Lazy frame index
// ─────────────────────────────────────────────────────────────────────────────

/// A lightweight index entry produced during lazy scanning.
///
/// Only the 10-byte frame header is decoded — the body bytes are not parsed
/// until `LazyId3v2::decode_frame` is called.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameIndexEntry {
    /// The 4-character frame ID (e.g. `"TIT2"`).
    pub frame_id: String,
    /// Byte offset of the frame *body* within the original tag data.
    pub body_offset: usize,
    /// Length of the frame body in bytes.
    pub body_len: usize,
    /// Raw 2-byte frame flags.
    pub flags: u16,
}

/// Lazy ID3v2 parser.
///
/// `LazyId3v2` scans only frame *headers* on construction, building an index.
/// Frame bodies are decoded on demand via [`LazyId3v2::decode_frame`].
///
/// This is useful when you only need a small number of tags from a large file
/// (e.g. extracting just the title and duration).
///
/// # Example
///
/// ```rust,no_run
/// use oximedia_metadata::id3v2::LazyId3v2;
///
/// // (In a real application `data` would come from reading a file.)
/// let data: Vec<u8> = Vec::new();
/// if let Ok(lazy) = LazyId3v2::new(&data) {
///     println!("Tag version: 2.{}", lazy.version());
///     for entry in lazy.index() {
///         println!("Frame: {}", entry.frame_id);
///     }
/// }
/// ```
pub struct LazyId3v2 {
    /// Original tag bytes (owns a copy so the caller is not tied to a lifetime).
    data: Vec<u8>,
    /// ID3v2 minor version (3 or 4).
    version: u8,
    /// Index of all frame headers, in order.
    index: Vec<FrameIndexEntry>,
}

impl LazyId3v2 {
    /// Build a lazy index from `data`.
    ///
    /// Only frame headers are decoded; no frame bodies are read.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` is not a valid ID3v2.3 or ID3v2.4 tag header.
    pub fn new(data: &[u8]) -> Result<Self, Error> {
        if data.len() < ID3V2_HEADER_SIZE {
            return Err(Error::ParseError(
                "Data too short for ID3v2 header".to_string(),
            ));
        }
        if &data[0..3] != b"ID3" {
            return Err(Error::ParseError("Not an ID3v2 tag".to_string()));
        }

        let version = data[3];
        let flags = data[5];

        if version != ID3V2_VERSION_3 && version != ID3V2_VERSION_4 {
            return Err(Error::Unsupported(format!("ID3v2.{version} not supported")));
        }

        let tag_size = decode_synchsafe_int(&data[6..10])? as usize;

        // Skip extended header if present
        let mut frame_start = ID3V2_HEADER_SIZE;
        if flags & 0x40 != 0 {
            if data.len() < frame_start + 4 {
                return Err(Error::ParseError("Extended header too short".to_string()));
            }
            let ext_size = if version == ID3V2_VERSION_4 {
                decode_synchsafe_int(&data[frame_start..frame_start + 4])? as usize
            } else {
                u32::from_be_bytes([
                    data[frame_start],
                    data[frame_start + 1],
                    data[frame_start + 2],
                    data[frame_start + 3],
                ]) as usize
            };
            frame_start += ext_size;
        }

        let tag_end = ID3V2_HEADER_SIZE + tag_size;
        if data.len() < tag_end {
            return Err(Error::ParseError(
                "Tag size exceeds data length".to_string(),
            ));
        }

        // Scan frame headers only
        let mut index = Vec::new();
        let mut pos = frame_start;

        while pos + ID3V2_FRAME_HEADER_SIZE <= tag_end {
            // Padding detection
            if data[pos] == 0 {
                break;
            }

            let frame_id = std::str::from_utf8(&data[pos..pos + 4])
                .map_err(|e| Error::ParseError(format!("Invalid frame ID: {e}")))?
                .to_string();

            let frame_size = if version == ID3V2_VERSION_4 {
                decode_synchsafe_int(&data[pos + 4..pos + 8])? as usize
            } else {
                u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                    as usize
            };

            let frame_flags = u16::from_be_bytes([data[pos + 8], data[pos + 9]]);

            let body_offset = pos + ID3V2_FRAME_HEADER_SIZE;

            if body_offset + frame_size > tag_end {
                return Err(Error::ParseError("Frame size exceeds tag size".to_string()));
            }

            index.push(FrameIndexEntry {
                frame_id,
                body_offset,
                body_len: frame_size,
                flags: frame_flags,
            });

            pos = body_offset + frame_size;
        }

        Ok(Self {
            data: data.to_vec(),
            version,
            index,
        })
    }

    /// Return the ID3v2 minor version (3 or 4).
    #[must_use]
    pub fn version(&self) -> u8 {
        self.version
    }

    /// Return the complete frame index (header-only entries).
    #[must_use]
    pub fn index(&self) -> &[FrameIndexEntry] {
        &self.index
    }

    /// Return all index entries whose `frame_id` matches `id`.
    #[must_use]
    pub fn find(&self, id: &str) -> Vec<&FrameIndexEntry> {
        self.index.iter().filter(|e| e.frame_id == id).collect()
    }

    /// Decode (parse) the body of the first frame whose ID matches `id`.
    ///
    /// Returns `None` if no such frame exists in the index.
    ///
    /// # Errors
    ///
    /// Returns an error if the body bytes are malformed.
    pub fn decode_frame(&self, id: &str) -> Result<Option<MetadataValue>, Error> {
        let entry = match self.index.iter().find(|e| e.frame_id == id) {
            Some(e) => e,
            None => return Ok(None),
        };
        let body = &self.data[entry.body_offset..entry.body_offset + entry.body_len];
        let value = parse_frame(id, body)?;
        Ok(Some(value))
    }

    /// Decode the body of every frame in the index into a full `Metadata`.
    ///
    /// This is equivalent to `parse()` but uses the pre-built index.
    ///
    /// # Errors
    ///
    /// Returns an error if any frame body is malformed.
    pub fn decode_all(&self) -> Result<Metadata, Error> {
        let mut metadata = Metadata::new(MetadataFormat::Id3v2);
        for entry in &self.index {
            let body = &self.data[entry.body_offset..entry.body_offset + entry.body_len];
            let value = parse_frame(&entry.frame_id, body)?;
            metadata.insert(entry.frame_id.clone(), value);
        }
        Ok(metadata)
    }

    /// Return the raw (unparsed) body bytes of a frame by ID.
    ///
    /// Useful for inspecting or forwarding frames without full decoding.
    #[must_use]
    pub fn raw_body(&self, id: &str) -> Option<&[u8]> {
        self.index
            .iter()
            .find(|e| e.frame_id == id)
            .map(|e| &self.data[e.body_offset..e.body_offset + e.body_len])
    }

    /// Count the total number of indexed frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.index.len()
    }

    /// Return `true` if the index contains a frame with the given ID.
    #[must_use]
    pub fn contains_frame(&self, id: &str) -> bool {
        self.index.iter().any(|e| e.frame_id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synchsafe_int() {
        let encoded = encode_synchsafe_int(1000);
        assert_eq!(
            decode_synchsafe_int(&encoded).expect("should succeed in test"),
            1000
        );

        let encoded = encode_synchsafe_int(0);
        assert_eq!(
            decode_synchsafe_int(&encoded).expect("should succeed in test"),
            0
        );

        let encoded = encode_synchsafe_int(268435455); // Max synchsafe value
        assert_eq!(
            decode_synchsafe_int(&encoded).expect("should succeed in test"),
            268435455
        );
    }

    #[test]
    fn test_text_encoding() {
        let text = "Hello World";

        // Latin-1
        let encoded = TextEncoding::Latin1.encode(text);
        assert_eq!(
            TextEncoding::Latin1
                .decode(&encoded)
                .expect("should succeed in test"),
            text
        );

        // UTF-8
        let encoded = TextEncoding::Utf8.encode(text);
        assert_eq!(
            TextEncoding::Utf8
                .decode(&encoded)
                .expect("should succeed in test"),
            text
        );
    }

    #[test]
    fn test_decode_static_latin1_idempotent() {
        // Decoding the same Latin-1 bytes twice must yield identical Strings
        // (guards the reused-decoder path against state leaking between calls).
        let bytes = b"Hello World";
        let first = decode_static(WINDOWS_1252, bytes);
        let second = decode_static(WINDOWS_1252, bytes);
        assert_eq!(first, second);
        assert_eq!(first, "Hello World");
    }

    #[test]
    fn test_decode_static_latin1_golden() {
        // Windows-1252 (ID3v2 "Latin-1"): 0xE9 -> 'é', 0xC0 -> 'À'.
        // Fixed golden mapping for a known byte sequence.
        let bytes = [0x43, 0x61, 0x66, 0xE9, 0x20, 0xC0]; // "Café À"
        let decoded = decode_static(WINDOWS_1252, &bytes);
        assert_eq!(decoded, "Caf\u{00E9} \u{00C0}");
    }

    #[test]
    fn test_decode_static_utf8_golden() {
        let text = "Stra\u{00DF}e \u{1F3B5}"; // "Straße 🎵"
        let decoded = decode_static(UTF_8, text.as_bytes());
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_decode_static_matches_convenience_latin1() {
        // The reused-decoder helper must agree byte-for-byte with the previous
        // convenience `Encoding::decode` path for a full Latin-1 range sweep.
        let bytes: Vec<u8> = (0u8..=255).collect();
        let via_helper = decode_static(WINDOWS_1252, &bytes);
        let (via_convenience, _, _) = WINDOWS_1252.decode(&bytes);
        assert_eq!(via_helper, via_convenience.into_owned());
    }

    #[test]
    fn test_parse_text_frame_utf16_with_bom_le() {
        // UTF-16LE BOM (FF FE) + "Hi" -> H=0x48 i=0x69, both as LE code units.
        let mut data = vec![1u8]; // encoding byte 1 = UTF-16 (BOM-dependent)
        data.extend_from_slice(&[0xFF, 0xFE]); // LE BOM
        data.extend_from_slice(&[0x48, 0x00, 0x69, 0x00]); // "Hi"
        let value = parse_text_frame(&data).expect("decode should succeed");
        assert_eq!(value.as_text(), Some("Hi"));
    }

    #[test]
    fn test_parse_text_frame_utf16_with_bom_be() {
        // UTF-16BE BOM (FE FF) + "Hi" as BE code units; guards BOM-endianness
        // selection survives the reused-decoder rewrite.
        let mut data = vec![1u8]; // encoding byte 1 = UTF-16 (BOM-dependent)
        data.extend_from_slice(&[0xFE, 0xFF]); // BE BOM
        data.extend_from_slice(&[0x00, 0x48, 0x00, 0x69]); // "Hi"
        let value = parse_text_frame(&data).expect("decode should succeed");
        assert_eq!(value.as_text(), Some("Hi"));
    }

    #[test]
    fn test_parse_text_frame_utf16_bom_roundtrip_unicode() {
        // Write a UTF-16 (BOM) frame then parse it back; the BOM-aware path must
        // reproduce the original non-ASCII text exactly.
        let text = "M\u{00FC}sic \u{2122}"; // "Müsic ™"
        let frame_body = write_text_frame(text, TextEncoding::Utf16);
        let value = parse_text_frame(&frame_body).expect("decode should succeed");
        assert_eq!(value.as_text(), Some(text));
    }

    #[test]
    fn test_text_encoding_byte() {
        assert_eq!(TextEncoding::Latin1.to_byte(), 0);
        assert_eq!(TextEncoding::Utf16.to_byte(), 1);
        assert_eq!(TextEncoding::Utf16Be.to_byte(), 2);
        assert_eq!(TextEncoding::Utf8.to_byte(), 3);

        assert_eq!(TextEncoding::from_byte(0), TextEncoding::Latin1);
        assert_eq!(TextEncoding::from_byte(1), TextEncoding::Utf16);
        assert_eq!(TextEncoding::from_byte(2), TextEncoding::Utf16Be);
        assert_eq!(TextEncoding::from_byte(3), TextEncoding::Utf8);
    }

    #[test]
    fn test_write_text_frame_utf8_default() {
        let text = "Test Title";
        let data = write_text_frame(text, TextEncoding::Utf8);

        assert_eq!(data[0], TextEncoding::Utf8.to_byte());
        assert_eq!(&data[1..], text.as_bytes());
    }

    #[test]
    fn test_default_write_encoding_is_utf8() {
        assert_eq!(Id3v2WriteEncoding::default(), Id3v2WriteEncoding::Utf8);
    }

    #[test]
    fn test_default_write_options_v24_utf8() {
        let opts = Id3v2WriteOptions::default();
        assert_eq!(opts.version, ID3V2_VERSION_4);
        assert_eq!(opts.encoding, Id3v2WriteEncoding::Utf8);
    }

    #[test]
    fn test_write_v24_utf8_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::Id3v2);
        metadata.insert(
            "TIT2".to_string(),
            MetadataValue::Text("Hello UTF-8".to_string()),
        );
        metadata.insert(
            "TPE1".to_string(),
            MetadataValue::Text("Artist".to_string()),
        );

        let opts = Id3v2WriteOptions {
            version: ID3V2_VERSION_4,
            encoding: Id3v2WriteEncoding::Utf8,
        };
        let data = write_with_options(&metadata, &opts).expect("write should succeed");
        let parsed = parse(&data).expect("parse should succeed");

        assert_eq!(
            parsed.get("TIT2").and_then(|v| v.as_text()),
            Some("Hello UTF-8")
        );
        assert_eq!(parsed.get("TPE1").and_then(|v| v.as_text()), Some("Artist"));
    }

    #[test]
    fn test_write_v24_latin1_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::Id3v2);
        metadata.insert(
            "TIT2".to_string(),
            MetadataValue::Text("Latin Title".to_string()),
        );

        let opts = Id3v2WriteOptions {
            version: ID3V2_VERSION_4,
            encoding: Id3v2WriteEncoding::Latin1,
        };
        let data = write_with_options(&metadata, &opts).expect("write should succeed");
        let parsed = parse(&data).expect("parse should succeed");

        assert_eq!(
            parsed.get("TIT2").and_then(|v| v.as_text()),
            Some("Latin Title")
        );
    }

    #[test]
    fn test_write_v23_falls_back_from_utf8_to_utf16() {
        // v2.3 does not support UTF-8; the writer should automatically fall back.
        let mut metadata = Metadata::new(MetadataFormat::Id3v2);
        metadata.insert(
            "TIT2".to_string(),
            MetadataValue::Text("Fallback".to_string()),
        );

        let opts = Id3v2WriteOptions {
            version: ID3V2_VERSION_3,
            encoding: Id3v2WriteEncoding::Utf8, // should auto-downgrade
        };
        let data = write_with_options(&metadata, &opts).expect("write should succeed");

        // The tag header should say v2.3
        assert_eq!(data[3], ID3V2_VERSION_3);

        // Parse the v2.3 tag (our parser handles both versions)
        let parsed = parse(&data).expect("parse should succeed");
        assert_eq!(
            parsed.get("TIT2").and_then(|v| v.as_text()),
            Some("Fallback")
        );
    }

    #[test]
    fn test_write_integer_value_as_text_frame() {
        let mut metadata = Metadata::new(MetadataFormat::Id3v2);
        metadata.insert("TRCK".to_string(), MetadataValue::Integer(7));

        let data = write(&metadata).expect("write should succeed");
        let parsed = parse(&data).expect("parse should succeed");
        assert_eq!(parsed.get("TRCK").and_then(|v| v.as_text()), Some("7"));
    }

    #[test]
    fn test_write_encoding_enum_coverage() {
        assert_eq!(
            Id3v2WriteEncoding::Utf8.to_text_encoding(),
            TextEncoding::Utf8
        );
        assert_eq!(
            Id3v2WriteEncoding::Utf16.to_text_encoding(),
            TextEncoding::Utf16
        );
        assert_eq!(
            Id3v2WriteEncoding::Latin1.to_text_encoding(),
            TextEncoding::Latin1
        );
    }

    #[test]
    fn test_write_v24_synchsafe_frame_sizes() {
        let mut metadata = Metadata::new(MetadataFormat::Id3v2);
        metadata.insert(
            "TIT2".to_string(),
            MetadataValue::Text("Synchsafe".to_string()),
        );

        let opts = Id3v2WriteOptions {
            version: ID3V2_VERSION_4,
            encoding: Id3v2WriteEncoding::Utf8,
        };
        let data = write_with_options(&metadata, &opts).expect("write should succeed");

        // Find the frame header: after 10-byte tag header
        // Frame header: 4-byte id + 4-byte size + 2-byte flags = 10 bytes
        // For v2.4, the frame size bytes should be synchsafe
        let frame_size_bytes = &data[14..18];
        // All synchsafe bytes have bit 7 clear
        for &b in frame_size_bytes {
            assert_eq!(b & 0x80, 0, "synchsafe byte should have bit 7 clear");
        }
    }

    #[test]
    fn test_parse_v24_utf8_text_frame() {
        // Build a minimal v2.4 tag with a UTF-8 text frame manually
        let text = "Unicode: \u{00E9}\u{00E8}\u{00EA}"; // e with accents
        let mut frame_body = vec![3u8]; // encoding = UTF-8
        frame_body.extend_from_slice(text.as_bytes());

        let frame_size = frame_body.len() as u32;
        let mut tag_body = Vec::new();
        tag_body.extend_from_slice(b"TIT2");
        tag_body.extend_from_slice(&encode_synchsafe_int(frame_size));
        tag_body.extend_from_slice(&[0u8; 2]); // flags
        tag_body.extend_from_slice(&frame_body);

        let mut data = Vec::new();
        data.extend_from_slice(b"ID3");
        data.push(4); // v2.4
        data.push(0);
        data.push(0);
        data.extend_from_slice(&encode_synchsafe_int(tag_body.len() as u32));
        data.extend_from_slice(&tag_body);

        let parsed = parse(&data).expect("parse should succeed");
        assert_eq!(parsed.get("TIT2").and_then(|v| v.as_text()), Some(text));
    }

    // ─── LazyId3v2 ────────────────────────────────────────────────────────

    /// Build a minimal v2.4 ID3 tag with the given frames.
    fn make_v24_tag(frames: &[(&str, &str)]) -> Vec<u8> {
        let mut body = Vec::new();
        for (id, text) in frames {
            let frame_body = write_text_frame(text, TextEncoding::Utf8);
            let sz = encode_synchsafe_int(frame_body.len() as u32);
            body.extend_from_slice(id.as_bytes());
            body.extend_from_slice(&sz);
            body.extend_from_slice(&[0u8; 2]); // flags
            body.extend_from_slice(&frame_body);
        }
        let mut tag = Vec::new();
        tag.extend_from_slice(b"ID3");
        tag.push(4); // v2.4
        tag.push(0); // revision
        tag.push(0); // flags
        tag.extend_from_slice(&encode_synchsafe_int(body.len() as u32));
        tag.extend_from_slice(&body);
        tag
    }

    #[test]
    fn test_lazy_id3v2_new_valid_tag() {
        let data = make_v24_tag(&[("TIT2", "Lazy Title"), ("TPE1", "Lazy Artist")]);
        let lazy = LazyId3v2::new(&data).expect("Should parse");
        assert_eq!(lazy.version(), 4);
        assert_eq!(lazy.frame_count(), 2);
    }

    #[test]
    fn test_lazy_id3v2_too_short() {
        let result = LazyId3v2::new(b"ID3");
        assert!(result.is_err());
    }

    #[test]
    fn test_lazy_id3v2_not_id3() {
        let mut data = vec![0u8; 20];
        data[0] = b'O';
        data[1] = b'G';
        data[2] = b'G';
        let result = LazyId3v2::new(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_lazy_id3v2_unsupported_version() {
        let mut data = vec![0u8; 20];
        data[0] = b'I';
        data[1] = b'D';
        data[2] = b'3';
        data[3] = 2; // v2.2 (unsupported)
        let result = LazyId3v2::new(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_lazy_id3v2_index_entries() {
        let data = make_v24_tag(&[("TIT2", "Song"), ("TALB", "Album")]);
        let lazy = LazyId3v2::new(&data).expect("Should parse");

        let ids: Vec<&str> = lazy.index().iter().map(|e| e.frame_id.as_str()).collect();
        assert!(ids.contains(&"TIT2"));
        assert!(ids.contains(&"TALB"));
    }

    #[test]
    fn test_lazy_id3v2_find() {
        let data = make_v24_tag(&[("TIT2", "Title"), ("TPE1", "Artist")]);
        let lazy = LazyId3v2::new(&data).expect("Should parse");

        let found = lazy.find("TIT2");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].frame_id, "TIT2");

        let not_found = lazy.find("TCOM");
        assert!(not_found.is_empty());
    }

    #[test]
    fn test_lazy_id3v2_contains_frame() {
        let data = make_v24_tag(&[("TIT2", "Title")]);
        let lazy = LazyId3v2::new(&data).expect("Should parse");

        assert!(lazy.contains_frame("TIT2"));
        assert!(!lazy.contains_frame("TCOM"));
    }

    #[test]
    fn test_lazy_id3v2_decode_frame_existing() {
        let data = make_v24_tag(&[("TIT2", "Decoded Title")]);
        let lazy = LazyId3v2::new(&data).expect("Should parse");

        let value = lazy
            .decode_frame("TIT2")
            .expect("Decode should succeed")
            .expect("Frame should exist");

        assert_eq!(value.as_text(), Some("Decoded Title"));
    }

    #[test]
    fn test_lazy_id3v2_decode_frame_missing() {
        let data = make_v24_tag(&[("TIT2", "Title")]);
        let lazy = LazyId3v2::new(&data).expect("Should parse");

        let value = lazy
            .decode_frame("TCOM")
            .expect("Decode of missing frame should not error");
        assert!(value.is_none());
    }

    #[test]
    fn test_lazy_id3v2_decode_all() {
        let data = make_v24_tag(&[("TIT2", "All Title"), ("TPE1", "All Artist")]);
        let lazy = LazyId3v2::new(&data).expect("Should parse");

        let metadata = lazy.decode_all().expect("Decode all should succeed");
        assert_eq!(
            metadata.get("TIT2").and_then(|v| v.as_text()),
            Some("All Title")
        );
        assert_eq!(
            metadata.get("TPE1").and_then(|v| v.as_text()),
            Some("All Artist")
        );
    }

    #[test]
    fn test_lazy_id3v2_raw_body() {
        let text = "Raw Body Test";
        let data = make_v24_tag(&[("TIT2", text)]);
        let lazy = LazyId3v2::new(&data).expect("Should parse");

        let raw = lazy.raw_body("TIT2").expect("Raw body should exist");
        // First byte is encoding (UTF-8 = 3), rest is text
        assert_eq!(raw[0], 3);
        assert_eq!(&raw[1..], text.as_bytes());
    }

    #[test]
    fn test_lazy_id3v2_raw_body_missing() {
        let data = make_v24_tag(&[("TIT2", "Title")]);
        let lazy = LazyId3v2::new(&data).expect("Should parse");
        assert!(lazy.raw_body("TCON").is_none());
    }

    #[test]
    fn test_lazy_id3v2_decode_all_matches_parse() {
        let data = make_v24_tag(&[
            ("TIT2", "Test Song"),
            ("TPE1", "Test Artist"),
            ("TALB", "Test Album"),
        ]);

        let eager = parse(&data).expect("Eager parse should succeed");
        let lazy = LazyId3v2::new(&data).expect("Lazy parse should succeed");
        let lazy_all = lazy.decode_all().expect("Decode all should succeed");

        // Both should produce the same results
        for frame_id in &["TIT2", "TPE1", "TALB"] {
            assert_eq!(
                eager.get(*frame_id).and_then(|v| v.as_text()),
                lazy_all.get(*frame_id).and_then(|v| v.as_text()),
                "Mismatch for frame {frame_id}"
            );
        }
    }

    #[test]
    fn test_lazy_id3v2_frame_index_entry_fields() {
        let text = "Header Fields";
        let data = make_v24_tag(&[("TIT2", text)]);
        let lazy = LazyId3v2::new(&data).expect("Should parse");

        let entry = &lazy.index()[0];
        assert_eq!(entry.frame_id, "TIT2");
        assert_eq!(entry.flags, 0); // no flags set in our test builder
                                    // body_len = 1 (encoding byte) + text.len()
        assert_eq!(entry.body_len, 1 + text.len());
        // body_offset > 10 (after the 10-byte tag header)
        assert!(entry.body_offset > 10);
    }

    #[test]
    fn test_lazy_id3v2_empty_tag_no_frames() {
        // Build a valid tag with no frames (just the header)
        let mut data = Vec::new();
        data.extend_from_slice(b"ID3");
        data.push(4); // v2.4
        data.push(0);
        data.push(0);
        data.extend_from_slice(&encode_synchsafe_int(0));

        let lazy = LazyId3v2::new(&data).expect("Should parse empty tag");
        assert_eq!(lazy.frame_count(), 0);
        assert!(lazy.index().is_empty());
    }

    #[test]
    fn test_lazy_id3v2_v23_support() {
        // Build a v2.3 tag (plain big-endian frame sizes)
        let text = "V23 Title";
        let frame_body = write_text_frame(text, TextEncoding::Utf16);
        let mut tag_body = Vec::new();
        tag_body.extend_from_slice(b"TIT2");
        tag_body.extend_from_slice(&(frame_body.len() as u32).to_be_bytes());
        tag_body.extend_from_slice(&[0u8; 2]);
        tag_body.extend_from_slice(&frame_body);

        let mut data = Vec::new();
        data.extend_from_slice(b"ID3");
        data.push(3); // v2.3
        data.push(0);
        data.push(0);
        data.extend_from_slice(&encode_synchsafe_int(tag_body.len() as u32));
        data.extend_from_slice(&tag_body);

        let lazy = LazyId3v2::new(&data).expect("v2.3 tag should parse");
        assert_eq!(lazy.version(), 3);
        assert_eq!(lazy.frame_count(), 1);

        let value = lazy
            .decode_frame("TIT2")
            .expect("Decode should succeed")
            .expect("Frame should exist");
        assert_eq!(value.as_text(), Some(text));
    }
}
