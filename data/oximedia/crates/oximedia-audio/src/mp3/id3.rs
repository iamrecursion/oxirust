//! ID3 tag parsing for MP3 files.
//!
//! This module implements ID3v1 and ID3v2 tag parsing for metadata extraction.

use crate::{AudioError, AudioResult};
use std::collections::HashMap;

/// ID3 tag information.
#[derive(Clone, Debug, Default)]
pub struct Id3Tag {
    /// Tag version.
    pub version: Id3Version,
    /// Title.
    pub title: Option<String>,
    /// Artist.
    pub artist: Option<String>,
    /// Album.
    pub album: Option<String>,
    /// Year.
    pub year: Option<String>,
    /// Comment.
    pub comment: Option<String>,
    /// Track number.
    pub track: Option<u8>,
    /// Genre.
    pub genre: Option<String>,
    /// Additional frames (ID3v2).
    pub frames: HashMap<String, Vec<u8>>,
}

/// ID3 version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Id3Version {
    /// No ID3 tag.
    #[default]
    None,
    /// ID3v1.
    V1,
    /// ID3v1.1 (with track number).
    V11,
    /// ID3v2.2.
    V22,
    /// ID3v2.3.
    V23,
    /// ID3v2.4.
    V24,
}

impl Id3Tag {
    /// Parse ID3v1 tag from end of file.
    ///
    /// # Errors
    ///
    /// Returns error if data is invalid.
    pub fn parse_v1(data: &[u8]) -> AudioResult<Self> {
        if data.len() < 128 {
            return Err(AudioError::InvalidData("ID3v1 tag too short".into()));
        }

        // Check for "TAG" identifier
        if &data[0..3] != b"TAG" {
            return Err(AudioError::InvalidData("Invalid ID3v1 tag".into()));
        }

        let mut tag = Self::default();

        // Parse fields (fixed size, null-padded)
        tag.title = parse_string(&data[3..33]);
        tag.artist = parse_string(&data[33..63]);
        tag.album = parse_string(&data[63..93]);
        tag.year = parse_string(&data[93..97]);

        // Check for ID3v1.1 (track number)
        if data[125] == 0 && data[126] != 0 {
            // ID3v1.1
            tag.comment = parse_string(&data[97..125]);
            tag.track = Some(data[126]);
            tag.version = Id3Version::V11;
        } else {
            // ID3v1.0
            tag.comment = parse_string(&data[97..127]);
            tag.version = Id3Version::V1;
        }

        // Parse genre
        let genre_id = data[127];
        tag.genre = get_genre(genre_id);

        Ok(tag)
    }

    /// Parse ID3v2 tag from beginning of file.
    ///
    /// # Errors
    ///
    /// Returns error if data is invalid.
    pub fn parse_v2(data: &[u8]) -> AudioResult<(Self, usize)> {
        if data.len() < 10 {
            return Err(AudioError::InvalidData("ID3v2 header too short".into()));
        }

        // Check for "ID3" identifier
        if &data[0..3] != b"ID3" {
            return Err(AudioError::InvalidData("Invalid ID3v2 tag".into()));
        }

        let major_version = data[3];
        let _minor_version = data[4];
        let flags = data[5];

        // Parse size (synchsafe integer)
        let size = synchsafe_to_u32(&data[6..10]) as usize;

        let version = match major_version {
            2 => Id3Version::V22,
            3 => Id3Version::V23,
            4 => Id3Version::V24,
            _ => {
                return Err(AudioError::UnsupportedFormat(format!(
                    "ID3v2.{major_version}"
                )))
            }
        };

        let mut tag = Self {
            version,
            ..Default::default()
        };

        // Parse extended header if present
        let mut offset = 10;
        if (flags & 0x40) != 0 {
            // Extended header present
            if data.len() < offset + 4 {
                return Err(AudioError::InvalidData("Extended header too short".into()));
            }
            let ext_size = synchsafe_to_u32(&data[offset..offset + 4]) as usize;
            offset += ext_size;
        }

        // Parse frames
        while offset + 10 < 10 + size && offset < data.len() {
            // Check for padding (all zeros)
            if data[offset] == 0 {
                break;
            }

            // Parse frame header
            let frame_id = match major_version {
                2 => {
                    // ID3v2.2 uses 3-character frame IDs
                    if offset + 6 > data.len() {
                        break;
                    }
                    String::from_utf8_lossy(&data[offset..offset + 3]).to_string()
                }
                _ => {
                    // ID3v2.3/2.4 use 4-character frame IDs
                    if offset + 10 > data.len() {
                        break;
                    }
                    String::from_utf8_lossy(&data[offset..offset + 4]).to_string()
                }
            };

            let (frame_size, frame_flags, header_size) = if major_version == 2 {
                // ID3v2.2: 3 bytes size, no flags
                let size = u32::from(data[offset + 3]) << 16
                    | u32::from(data[offset + 4]) << 8
                    | u32::from(data[offset + 5]);
                (size as usize, 0u16, 6)
            } else if major_version == 4 {
                // ID3v2.4: synchsafe size
                let size = synchsafe_to_u32(&data[offset + 4..offset + 8]) as usize;
                let flags = u16::from(data[offset + 8]) << 8 | u16::from(data[offset + 9]);
                (size, flags, 10)
            } else {
                // ID3v2.3: regular size
                let size = u32::from(data[offset + 4]) << 24
                    | u32::from(data[offset + 5]) << 16
                    | u32::from(data[offset + 6]) << 8
                    | u32::from(data[offset + 7]);
                let flags = u16::from(data[offset + 8]) << 8 | u16::from(data[offset + 9]);
                (size as usize, flags, 10)
            };

            offset += header_size;

            if offset + frame_size > data.len() {
                break;
            }

            // Parse frame data
            let frame_data = &data[offset..offset + frame_size];
            tag.parse_frame(&frame_id, frame_data, frame_flags)?;

            offset += frame_size;
        }

        Ok((tag, 10 + size))
    }

    /// Parse individual ID3v2 frame.
    fn parse_frame(&mut self, id: &str, data: &[u8], _flags: u16) -> AudioResult<()> {
        if data.is_empty() {
            return Ok(());
        }

        // Text frames start with encoding byte
        let encoding = data[0];
        let text_data = &data[1..];

        match id {
            "TIT2" | "TT2" => {
                // Title
                self.title = decode_text(text_data, encoding);
            }
            "TPE1" | "TP1" => {
                // Artist
                self.artist = decode_text(text_data, encoding);
            }
            "TALB" | "TAL" => {
                // Album
                self.album = decode_text(text_data, encoding);
            }
            "TYER" | "TYE" | "TDRC" => {
                // Year
                self.year = decode_text(text_data, encoding);
            }
            "COMM" | "COM" => {
                // Comment (skip language and description)
                if text_data.len() > 4 {
                    self.comment = decode_text(&text_data[4..], encoding);
                }
            }
            "TRCK" | "TRK" => {
                // Track number
                if let Some(track_str) = decode_text(text_data, encoding) {
                    // Parse "1" or "1/12" format
                    if let Some(num_str) = track_str.split('/').next() {
                        if let Ok(track) = num_str.trim().parse::<u8>() {
                            self.track = Some(track);
                        }
                    }
                }
            }
            "TCON" | "TCO" => {
                // Genre
                self.genre = decode_text(text_data, encoding);
            }
            _ => {
                // Store unknown frames
                self.frames.insert(id.to_string(), data.to_vec());
            }
        }

        Ok(())
    }

    /// Get tag size from file data.
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn get_tag_size(data: &[u8]) -> AudioResult<usize> {
        if data.len() < 10 {
            return Ok(0);
        }

        if &data[0..3] != b"ID3" {
            return Ok(0);
        }

        let size = synchsafe_to_u32(&data[6..10]) as usize;
        Ok(10 + size)
    }
}

/// Parse null-terminated or null-padded string.
fn parse_string(data: &[u8]) -> Option<String> {
    let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
    let trimmed = &data[..end];

    if trimmed.is_empty() {
        return None;
    }

    let s = String::from_utf8_lossy(trimmed).trim().to_string();

    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Decode text with specified encoding.
fn decode_text(data: &[u8], encoding: u8) -> Option<String> {
    if data.is_empty() {
        return None;
    }

    let s = match encoding {
        0 => {
            // ISO-8859-1
            let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
            String::from_utf8_lossy(&data[..end]).to_string()
        }
        1 => {
            // UTF-16 with BOM
            decode_utf16(data)
        }
        2 => {
            // UTF-16BE
            decode_utf16_be(data)
        }
        3 => {
            // UTF-8
            let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
            String::from_utf8_lossy(&data[..end]).to_string()
        }
        _ => String::from_utf8_lossy(data).to_string(),
    };

    let trimmed = s.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

/// Decode UTF-16 with BOM.
fn decode_utf16(data: &[u8]) -> String {
    if data.len() < 2 {
        return String::new();
    }

    // Check BOM
    let le = data[0] == 0xFF && data[1] == 0xFE;
    let be = data[0] == 0xFE && data[1] == 0xFF;

    let start = if le || be { 2 } else { 0 };

    if le {
        decode_utf16_le(&data[start..])
    } else {
        decode_utf16_be(&data[start..])
    }
}

/// Decode UTF-16 little-endian.
fn decode_utf16_le(data: &[u8]) -> String {
    let mut chars = Vec::new();
    for chunk in data.chunks_exact(2) {
        let code = u16::from_le_bytes([chunk[0], chunk[1]]);
        if code == 0 {
            break;
        }
        chars.push(code);
    }
    String::from_utf16_lossy(&chars)
}

/// Decode UTF-16 big-endian.
fn decode_utf16_be(data: &[u8]) -> String {
    let mut chars = Vec::new();
    for chunk in data.chunks_exact(2) {
        let code = u16::from_be_bytes([chunk[0], chunk[1]]);
        if code == 0 {
            break;
        }
        chars.push(code);
    }
    String::from_utf16_lossy(&chars)
}

/// Convert synchsafe integer to u32.
fn synchsafe_to_u32(data: &[u8]) -> u32 {
    debug_assert!(data.len() >= 4);
    u32::from(data[0] & 0x7F) << 21
        | u32::from(data[1] & 0x7F) << 14
        | u32::from(data[2] & 0x7F) << 7
        | u32::from(data[3] & 0x7F)
}

/// Get genre name from ID3v1 genre ID.
fn get_genre(id: u8) -> Option<String> {
    GENRES.get(id as usize).map(|&s| s.to_string())
}

/// ID3v1 genre list.
const GENRES: &[&str] = &[
    "Blues",
    "Classic Rock",
    "Country",
    "Dance",
    "Disco",
    "Funk",
    "Grunge",
    "Hip-Hop",
    "Jazz",
    "Metal",
    "New Age",
    "Oldies",
    "Other",
    "Pop",
    "R&B",
    "Rap",
    "Reggae",
    "Rock",
    "Techno",
    "Industrial",
    "Alternative",
    "Ska",
    "Death Metal",
    "Pranks",
    "Soundtrack",
    "Euro-Techno",
    "Ambient",
    "Trip-Hop",
    "Vocal",
    "Jazz+Funk",
    "Fusion",
    "Trance",
    "Classical",
    "Instrumental",
    "Acid",
    "House",
    "Game",
    "Sound Clip",
    "Gospel",
    "Noise",
    "AlternRock",
    "Bass",
    "Soul",
    "Punk",
    "Space",
    "Meditative",
    "Instrumental Pop",
    "Instrumental Rock",
    "Ethnic",
    "Gothic",
    "Darkwave",
    "Techno-Industrial",
    "Electronic",
    "Pop-Folk",
    "Eurodance",
    "Dream",
    "Southern Rock",
    "Comedy",
    "Cult",
    "Gangsta",
    "Top 40",
    "Christian Rap",
    "Pop/Funk",
    "Jungle",
    "Native American",
    "Cabaret",
    "New Wave",
    "Psychedelic",
    "Rave",
    "Showtunes",
    "Trailer",
    "Lo-Fi",
    "Tribal",
    "Acid Punk",
    "Acid Jazz",
    "Polka",
    "Retro",
    "Musical",
    "Rock & Roll",
    "Hard Rock",
];
