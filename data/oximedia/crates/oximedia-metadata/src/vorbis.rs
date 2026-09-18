//! Vorbis Comments parsing and writing support.
//!
//! Vorbis Comments are used in Ogg Vorbis, FLAC, Opus, and other formats.
//!
//! # Format
//!
//! Vorbis Comments consist of:
//! - Vendor string (length-prefixed UTF-8 string)
//! - User comment list count (32-bit little-endian)
//! - User comments (each is a length-prefixed UTF-8 string in "NAME=value" format)
//!
//! # Common Fields
//!
//! - **TITLE**: Track title
//! - **ARTIST**: Artist name
//! - **ALBUM**: Album title
//! - **ALBUMARTIST**: Album artist
//! - **TRACKNUMBER**: Track number
//! - **DATE**: Release date
//! - **GENRE**: Genre

use crate::{Error, Metadata, MetadataFormat, MetadataValue};
use std::io::{Cursor, Read};

/// Vorbis Comment multi-value helper.
///
/// Per the Vorbis Comment specification, a field name MAY appear multiple
/// times.  Each occurrence represents an additional value for that field.
/// This struct provides ergonomic access to multi-value Vorbis Comment fields.
#[derive(Debug, Clone)]
pub struct VorbisComments {
    /// The underlying metadata container.
    inner: Metadata,
}

impl VorbisComments {
    /// Create from an existing `Metadata` (must be `MetadataFormat::VorbisComments`).
    pub fn new() -> Self {
        Self {
            inner: Metadata::new(MetadataFormat::VorbisComments),
        }
    }

    /// Wrap an already-parsed `Metadata` container.
    pub fn from_metadata(metadata: Metadata) -> Self {
        Self { inner: metadata }
    }

    /// Get the first value for a field (case-insensitive lookup).
    pub fn get_first(&self, key: &str) -> Option<&str> {
        let upper = key.to_uppercase();
        match self.inner.get(&upper) {
            Some(MetadataValue::Text(s)) => Some(s.as_str()),
            Some(MetadataValue::TextList(list)) => list.first().map(|s| s.as_str()),
            _ => None,
        }
    }

    /// Get all values for a field (case-insensitive lookup).
    ///
    /// Returns an empty slice if the field does not exist.
    pub fn get_all(&self, key: &str) -> Vec<&str> {
        let upper = key.to_uppercase();
        match self.inner.get(&upper) {
            Some(MetadataValue::Text(s)) => vec![s.as_str()],
            Some(MetadataValue::TextList(list)) => list.iter().map(|s| s.as_str()).collect(),
            _ => Vec::new(),
        }
    }

    /// Set a single value for a field, replacing any previous values.
    pub fn set(&mut self, key: &str, value: &str) {
        let upper = key.to_uppercase();
        self.inner
            .insert(upper, MetadataValue::Text(value.to_string()));
    }

    /// Append a value to a field, creating a multi-value entry.
    ///
    /// Per the Vorbis Comment specification, multiple values for the same
    /// field name are represented by repeated comment entries.
    pub fn add_value(&mut self, key: &str, value: &str) {
        let upper = key.to_uppercase();
        match self.inner.get(&upper).cloned() {
            Some(MetadataValue::Text(existing)) => {
                let list = vec![existing, value.to_string()];
                self.inner.insert(upper, MetadataValue::TextList(list));
            }
            Some(MetadataValue::TextList(mut list)) => {
                list.push(value.to_string());
                self.inner.insert(upper, MetadataValue::TextList(list));
            }
            _ => {
                self.inner
                    .insert(upper, MetadataValue::Text(value.to_string()));
            }
        }
    }

    /// Remove all values for a field (case-insensitive).
    pub fn remove_all(&mut self, key: &str) -> bool {
        let upper = key.to_uppercase();
        self.inner.remove(&upper).is_some()
    }

    /// Return the number of distinct field names (not counting repeated values).
    pub fn field_count(&self) -> usize {
        self.inner
            .fields()
            .keys()
            .filter(|k| k.as_str() != "VENDOR")
            .count()
    }

    /// Return the total number of comment entries (counting each repeated value).
    pub fn total_value_count(&self) -> usize {
        let mut count = 0usize;
        for (key, value) in self.inner.fields() {
            if key == "VENDOR" {
                continue;
            }
            match value {
                MetadataValue::Text(_) => count += 1,
                MetadataValue::TextList(list) => count += list.len(),
                _ => {}
            }
        }
        count
    }

    /// Set the vendor string.
    pub fn set_vendor(&mut self, vendor: &str) {
        self.inner.insert(
            "VENDOR".to_string(),
            MetadataValue::Text(vendor.to_string()),
        );
    }

    /// Get the vendor string.
    pub fn vendor(&self) -> Option<&str> {
        self.inner.get("VENDOR").and_then(|v| v.as_text())
    }

    /// Consume and return the inner `Metadata`.
    pub fn into_metadata(self) -> Metadata {
        self.inner
    }

    /// Borrow the inner `Metadata`.
    pub fn metadata(&self) -> &Metadata {
        &self.inner
    }
}

impl Default for VorbisComments {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse Vorbis Comments from data.
///
/// # Errors
///
/// Returns an error if the data is not valid Vorbis Comments.
pub fn parse(data: &[u8]) -> Result<Metadata, Error> {
    let mut cursor = Cursor::new(data);
    let mut metadata = Metadata::new(MetadataFormat::VorbisComments);

    // Read vendor string length
    let vendor_length = read_u32_le(&mut cursor)?;

    // Read vendor string
    let mut vendor_bytes = vec![0u8; vendor_length as usize];
    cursor
        .read_exact(&mut vendor_bytes)
        .map_err(|e| Error::ParseError(format!("Failed to read vendor string: {e}")))?;

    let vendor = String::from_utf8(vendor_bytes)
        .map_err(|e| Error::EncodingError(format!("Invalid UTF-8 in vendor string: {e}")))?;

    // Store vendor string
    metadata.insert("VENDOR".to_string(), MetadataValue::Text(vendor));

    // Read user comment list count
    let comment_count = read_u32_le(&mut cursor)?;

    // Read user comments
    for _ in 0..comment_count {
        let comment_length = read_u32_le(&mut cursor)?;

        let mut comment_bytes = vec![0u8; comment_length as usize];
        cursor
            .read_exact(&mut comment_bytes)
            .map_err(|e| Error::ParseError(format!("Failed to read comment: {e}")))?;

        let comment = String::from_utf8(comment_bytes)
            .map_err(|e| Error::EncodingError(format!("Invalid UTF-8 in comment: {e}")))?;

        // Parse "NAME=value" format
        if let Some(eq_pos) = comment.find('=') {
            let name = comment[..eq_pos].to_uppercase();
            let value = &comment[eq_pos + 1..];

            // Check if field already exists (for multi-value fields)
            if let Some(existing) = metadata.get(&name) {
                // Convert to list if not already
                match existing {
                    MetadataValue::Text(text) => {
                        let list = vec![text.clone(), value.to_string()];
                        metadata.insert(name, MetadataValue::TextList(list));
                    }
                    MetadataValue::TextList(list) => {
                        let mut new_list = list.clone();
                        new_list.push(value.to_string());
                        metadata.insert(name, MetadataValue::TextList(new_list));
                    }
                    _ => {}
                }
            } else {
                metadata.insert(name, MetadataValue::Text(value.to_string()));
            }
        }
    }

    Ok(metadata)
}

/// Write Vorbis Comments to bytes.
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn write(metadata: &Metadata) -> Result<Vec<u8>, Error> {
    let mut result = Vec::new();

    // Get vendor string (or use default)
    let vendor = metadata
        .get("VENDOR")
        .and_then(|v| v.as_text())
        .unwrap_or("OxiMedia");

    // Write vendor string length and data
    write_u32_le(&mut result, vendor.len() as u32);
    result.extend_from_slice(vendor.as_bytes());

    // Collect comments
    let mut comments = Vec::new();
    for (key, value) in metadata.fields() {
        if key == "VENDOR" {
            continue; // Skip vendor field
        }

        match value {
            MetadataValue::Text(text) => {
                let comment = format!("{key}={text}");
                comments.push(comment);
            }
            MetadataValue::TextList(list) => {
                for text in list {
                    let comment = format!("{key}={text}");
                    comments.push(comment);
                }
            }
            _ => {
                // Skip non-text values
            }
        }
    }

    // Write comment count
    write_u32_le(&mut result, comments.len() as u32);

    // Write comments
    for comment in comments {
        write_u32_le(&mut result, comment.len() as u32);
        result.extend_from_slice(comment.as_bytes());
    }

    Ok(result)
}

/// Read a 32-bit little-endian unsigned integer.
fn read_u32_le(cursor: &mut Cursor<&[u8]>) -> Result<u32, Error> {
    let mut bytes = [0u8; 4];
    cursor
        .read_exact(&mut bytes)
        .map_err(|e| Error::ParseError(format!("Failed to read u32: {e}")))?;
    Ok(u32::from_le_bytes(bytes))
}

/// Write a 32-bit little-endian unsigned integer.
fn write_u32_le(buffer: &mut Vec<u8>, value: u32) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vorbis_comments_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::VorbisComments);

        metadata.insert(
            "TITLE".to_string(),
            MetadataValue::Text("Test Title".to_string()),
        );
        metadata.insert(
            "ARTIST".to_string(),
            MetadataValue::Text("Test Artist".to_string()),
        );
        metadata.insert(
            "ALBUM".to_string(),
            MetadataValue::Text("Test Album".to_string()),
        );
        metadata.insert(
            "TRACKNUMBER".to_string(),
            MetadataValue::Text("5".to_string()),
        );
        metadata.insert("DATE".to_string(), MetadataValue::Text("2024".to_string()));

        // Write
        let data = write(&metadata).expect("Write failed");

        // Parse
        let parsed = parse(&data).expect("Parse failed");

        assert_eq!(
            parsed.get("TITLE").and_then(|v| v.as_text()),
            Some("Test Title")
        );
        assert_eq!(
            parsed.get("ARTIST").and_then(|v| v.as_text()),
            Some("Test Artist")
        );
        assert_eq!(
            parsed.get("ALBUM").and_then(|v| v.as_text()),
            Some("Test Album")
        );
        assert_eq!(
            parsed.get("TRACKNUMBER").and_then(|v| v.as_text()),
            Some("5")
        );
        assert_eq!(parsed.get("DATE").and_then(|v| v.as_text()), Some("2024"));
    }

    #[test]
    fn test_vorbis_comments_multivalue() {
        let mut metadata = Metadata::new(MetadataFormat::VorbisComments);

        let artists = vec!["Artist 1".to_string(), "Artist 2".to_string()];
        metadata.insert("ARTIST".to_string(), MetadataValue::TextList(artists));

        // Write
        let data = write(&metadata).expect("Write failed");

        // Parse
        let parsed = parse(&data).expect("Parse failed");

        let parsed_artists = parsed
            .get("ARTIST")
            .and_then(|v| v.as_text_list())
            .expect("Expected text list");

        assert_eq!(parsed_artists.len(), 2);
        assert_eq!(parsed_artists[0], "Artist 1");
        assert_eq!(parsed_artists[1], "Artist 2");
    }

    #[test]
    fn test_read_write_u32_le() {
        let mut buffer = Vec::new();
        write_u32_le(&mut buffer, 12345);

        let mut cursor = Cursor::new(buffer.as_slice());
        let value = read_u32_le(&mut cursor).expect("should succeed in test");

        assert_eq!(value, 12345);
    }

    #[test]
    fn test_empty_vorbis_comments() {
        let metadata = Metadata::new(MetadataFormat::VorbisComments);

        // Write
        let data = write(&metadata).expect("Write failed");

        // Parse
        let parsed = parse(&data).expect("Parse failed");

        // Should have vendor string
        assert!(parsed.get("VENDOR").is_some());
    }

    // ------- VorbisComments helper tests -------

    #[test]
    fn test_vorbis_comments_helper_new() {
        let vc = VorbisComments::new();
        assert_eq!(vc.field_count(), 0);
        assert_eq!(vc.total_value_count(), 0);
    }

    #[test]
    fn test_vorbis_comments_set_and_get_first() {
        let mut vc = VorbisComments::new();
        vc.set("title", "My Song");
        assert_eq!(vc.get_first("TITLE"), Some("My Song"));
        // Case-insensitive lookup
        assert_eq!(vc.get_first("title"), Some("My Song"));
    }

    #[test]
    fn test_vorbis_comments_add_value_creates_list() {
        let mut vc = VorbisComments::new();
        vc.add_value("ARTIST", "Alice");
        vc.add_value("ARTIST", "Bob");
        vc.add_value("ARTIST", "Charlie");

        let all = vc.get_all("ARTIST");
        assert_eq!(all.len(), 3);
        assert_eq!(all[0], "Alice");
        assert_eq!(all[1], "Bob");
        assert_eq!(all[2], "Charlie");

        // get_first returns the first value
        assert_eq!(vc.get_first("ARTIST"), Some("Alice"));

        // field_count = 1 (one field name), total_value_count = 3
        assert_eq!(vc.field_count(), 1);
        assert_eq!(vc.total_value_count(), 3);
    }

    #[test]
    fn test_vorbis_comments_set_replaces_list() {
        let mut vc = VorbisComments::new();
        vc.add_value("GENRE", "Rock");
        vc.add_value("GENRE", "Pop");
        assert_eq!(vc.get_all("GENRE").len(), 2);

        // set() should replace all values
        vc.set("GENRE", "Jazz");
        assert_eq!(vc.get_all("GENRE"), vec!["Jazz"]);
    }

    #[test]
    fn test_vorbis_comments_remove_all() {
        let mut vc = VorbisComments::new();
        vc.add_value("ARTIST", "A");
        vc.add_value("ARTIST", "B");
        assert!(vc.remove_all("ARTIST"));
        assert!(vc.get_all("ARTIST").is_empty());
        // Removing again returns false
        assert!(!vc.remove_all("ARTIST"));
    }

    #[test]
    fn test_vorbis_comments_vendor() {
        let mut vc = VorbisComments::new();
        vc.set_vendor("OxiMedia 0.1.2");
        assert_eq!(vc.vendor(), Some("OxiMedia 0.1.2"));

        // Vendor should NOT count as a field
        assert_eq!(vc.field_count(), 0);
    }

    #[test]
    fn test_vorbis_comments_from_metadata_round_trip() {
        let mut vc = VorbisComments::new();
        vc.set("TITLE", "Song Title");
        vc.add_value("ARTIST", "Artist 1");
        vc.add_value("ARTIST", "Artist 2");
        vc.set("DATE", "2025");
        vc.set_vendor("TestVendor");

        // Write -> parse -> wrap
        let data = write(vc.metadata()).expect("write should succeed");
        let parsed = parse(&data).expect("parse should succeed");
        let vc2 = VorbisComments::from_metadata(parsed);

        assert_eq!(vc2.get_first("TITLE"), Some("Song Title"));
        assert_eq!(vc2.get_all("ARTIST").len(), 2);
        assert_eq!(vc2.get_first("DATE"), Some("2025"));
        assert_eq!(vc2.vendor(), Some("TestVendor"));
    }

    #[test]
    fn test_vorbis_comments_into_metadata() {
        let mut vc = VorbisComments::new();
        vc.set("ALBUM", "My Album");
        let metadata = vc.into_metadata();
        assert_eq!(metadata.format(), MetadataFormat::VorbisComments);
        assert_eq!(
            metadata.get("ALBUM").and_then(|v| v.as_text()),
            Some("My Album")
        );
    }

    #[test]
    fn test_vorbis_comments_get_all_nonexistent() {
        let vc = VorbisComments::new();
        assert!(vc.get_all("NONEXISTENT").is_empty());
        assert_eq!(vc.get_first("NONEXISTENT"), None);
    }

    #[test]
    fn test_parse_multivalue_from_raw_data() {
        // Build raw Vorbis Comment data with repeated ARTIST keys
        let mut raw = Vec::new();

        // Vendor string
        let vendor = b"TestEncoder";
        write_u32_le(&mut raw, vendor.len() as u32);
        raw.extend_from_slice(vendor);

        // 3 comments: ARTIST=Alice, ARTIST=Bob, TITLE=Song
        write_u32_le(&mut raw, 3);

        let c1 = b"ARTIST=Alice";
        write_u32_le(&mut raw, c1.len() as u32);
        raw.extend_from_slice(c1);

        let c2 = b"ARTIST=Bob";
        write_u32_le(&mut raw, c2.len() as u32);
        raw.extend_from_slice(c2);

        let c3 = b"TITLE=Song";
        write_u32_le(&mut raw, c3.len() as u32);
        raw.extend_from_slice(c3);

        let parsed = parse(&raw).expect("parse should succeed");
        let artists = parsed
            .get("ARTIST")
            .and_then(|v| v.as_text_list())
            .expect("should be a text list");
        assert_eq!(artists.len(), 2);
        assert_eq!(artists[0], "Alice");
        assert_eq!(artists[1], "Bob");
        assert_eq!(parsed.get("TITLE").and_then(|v| v.as_text()), Some("Song"));
    }

    #[test]
    fn test_vorbis_comments_default() {
        let vc = VorbisComments::default();
        assert_eq!(vc.field_count(), 0);
    }
}
