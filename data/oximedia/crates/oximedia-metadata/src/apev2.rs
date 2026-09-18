//! APEv2 tag parsing and writing support.
//!
//! APEv2 tags are used in APE (Monkey's Audio) files and sometimes in other formats.
//!
//! # Format
//!
//! APEv2 tags consist of:
//! - Header (32 bytes) with preamble "APETAGEX"
//! - Tag items (key-value pairs)
//! - Footer (32 bytes, same format as header)
//!
//! Each item has:
//! - Item value length (32-bit little-endian)
//! - Item flags (32-bit little-endian)
//! - Item key (null-terminated UTF-8 string)
//! - Item value (UTF-8 string or binary data)

use crate::{Error, Metadata, MetadataFormat, MetadataValue};
use std::io::{Cursor, Read};

/// APEv2 tag preamble
const APETAG_PREAMBLE: &[u8; 8] = b"APETAGEX";

/// APEv2 header/footer size
const APETAG_HEADER_SIZE: usize = 32;

/// APEv2 version 2.0
const APETAG_VERSION: u32 = 2000;

/// Item flag: Contains UTF-8 text
const ITEM_FLAG_TEXT: u32 = 0;

/// Item flag: Contains binary data
const ITEM_FLAG_BINARY: u32 = 1;

/// Parse APEv2 tags from data.
///
/// # Errors
///
/// Returns an error if the data is not a valid APEv2 tag.
pub fn parse(data: &[u8]) -> Result<Metadata, Error> {
    if data.len() < APETAG_HEADER_SIZE {
        return Err(Error::ParseError(
            "Data too short for APEv2 tag".to_string(),
        ));
    }

    // Check for footer (APEv2 tags typically have footer at the end)
    let footer_pos = data.len().saturating_sub(APETAG_HEADER_SIZE);
    if &data[footer_pos..footer_pos + 8] != APETAG_PREAMBLE {
        return Err(Error::ParseError("Not an APEv2 tag".to_string()));
    }

    let mut cursor = Cursor::new(&data[footer_pos..]);

    // Skip preamble
    cursor.set_position(8);

    // Read version
    let version = read_u32_le(&mut cursor)?;
    if version != APETAG_VERSION {
        return Err(Error::Unsupported(format!(
            "APEv2 version {version} not supported"
        )));
    }

    // Read tag size (excluding header/footer)
    let tag_size = read_u32_le(&mut cursor)?;

    // Read item count
    let item_count = read_u32_le(&mut cursor)?;

    // Read flags
    let _flags = read_u32_le(&mut cursor)?;

    // Calculate tag data position
    let tag_data_end = footer_pos;
    let tag_data_start = tag_data_end.saturating_sub(tag_size as usize);

    if tag_data_start >= data.len() {
        return Err(Error::ParseError("Invalid tag size".to_string()));
    }

    // Parse items
    let mut metadata = Metadata::new(MetadataFormat::Apev2);
    let mut pos = tag_data_start;

    for _ in 0..item_count {
        if pos + 8 > tag_data_end {
            break;
        }

        // Read item value length
        let value_length =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        // Read item flags
        let item_flags =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;

        // Read item key (null-terminated)
        let key_start = pos;
        while pos < tag_data_end && data[pos] != 0 {
            pos += 1;
        }

        if pos >= tag_data_end {
            return Err(Error::ParseError(
                "Item key not null-terminated".to_string(),
            ));
        }

        let key = String::from_utf8(data[key_start..pos].to_vec())
            .map_err(|e| Error::EncodingError(format!("Invalid UTF-8 in key: {e}")))?;
        pos += 1; // Skip null terminator

        // Read item value
        if pos + value_length > tag_data_end {
            return Err(Error::ParseError("Item value exceeds tag size".to_string()));
        }

        let value_data = &data[pos..pos + value_length];
        pos += value_length;

        // Determine value type based on flags
        let item_type = (item_flags >> 1) & 0x03;
        let value = if item_type == 1 {
            // Binary data
            MetadataValue::Binary(value_data.to_vec())
        } else {
            // UTF-8 text
            let text = String::from_utf8(value_data.to_vec())
                .map_err(|e| Error::EncodingError(format!("Invalid UTF-8 in value: {e}")))?;
            MetadataValue::Text(text)
        };

        metadata.insert(key, value);
    }

    Ok(metadata)
}

/// Write APEv2 tags to bytes.
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn write(metadata: &Metadata) -> Result<Vec<u8>, Error> {
    let mut items_data = Vec::new();
    let mut item_count: u32 = 0;

    // Write items
    for (key, value) in metadata.fields() {
        // Write item
        let (value_data, flags) = match value {
            MetadataValue::Text(text) => (text.as_bytes().to_vec(), ITEM_FLAG_TEXT << 1),
            MetadataValue::Binary(data) => (data.clone(), ITEM_FLAG_BINARY << 1),
            _ => continue, // Skip unsupported types
        };

        // Write item value length
        items_data.extend_from_slice(&(value_data.len() as u32).to_le_bytes());

        // Write item flags
        items_data.extend_from_slice(&flags.to_le_bytes());

        // Write item key (null-terminated)
        items_data.extend_from_slice(key.as_bytes());
        items_data.push(0);

        // Write item value
        items_data.extend_from_slice(&value_data);

        item_count += 1;
    }

    // Create footer
    let mut footer = Vec::new();

    // Write preamble
    footer.extend_from_slice(APETAG_PREAMBLE);

    // Write version
    footer.extend_from_slice(&APETAG_VERSION.to_le_bytes());

    // Write tag size (excluding footer)
    footer.extend_from_slice(&(items_data.len() as u32).to_le_bytes());

    // Write item count
    footer.extend_from_slice(&item_count.to_le_bytes());

    // Write flags (footer present, no header)
    footer.extend_from_slice(&0x8000_0000_u32.to_le_bytes());

    // Reserved (8 bytes)
    footer.extend_from_slice(&[0u8; 8]);

    // Combine items and footer
    let mut result = items_data;
    result.extend_from_slice(&footer);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apev2_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::Apev2);

        metadata.insert(
            "Title".to_string(),
            MetadataValue::Text("Test Title".to_string()),
        );
        metadata.insert(
            "Artist".to_string(),
            MetadataValue::Text("Test Artist".to_string()),
        );
        metadata.insert(
            "Album".to_string(),
            MetadataValue::Text("Test Album".to_string()),
        );
        metadata.insert("Track".to_string(), MetadataValue::Text("5".to_string()));

        // Write
        let data = write(&metadata).expect("Write failed");

        // Parse
        let parsed = parse(&data).expect("Parse failed");

        assert_eq!(
            parsed.get("Title").and_then(|v| v.as_text()),
            Some("Test Title")
        );
        assert_eq!(
            parsed.get("Artist").and_then(|v| v.as_text()),
            Some("Test Artist")
        );
        assert_eq!(
            parsed.get("Album").and_then(|v| v.as_text()),
            Some("Test Album")
        );
        assert_eq!(parsed.get("Track").and_then(|v| v.as_text()), Some("5"));
    }

    #[test]
    fn test_apev2_binary_data() {
        let mut metadata = Metadata::new(MetadataFormat::Apev2);

        let binary_data = vec![0x00, 0x01, 0x02, 0x03, 0xFF];
        metadata.insert(
            "Cover Art".to_string(),
            MetadataValue::Binary(binary_data.clone()),
        );

        // Write
        let data = write(&metadata).expect("Write failed");

        // Parse
        let parsed = parse(&data).expect("Parse failed");

        assert_eq!(
            parsed.get("Cover Art").and_then(|v| v.as_binary()),
            Some(binary_data.as_slice())
        );
    }

    #[test]
    fn test_apev2_empty() {
        let metadata = Metadata::new(MetadataFormat::Apev2);

        // Write
        let data = write(&metadata).expect("Write failed");

        // Parse
        let parsed = parse(&data).expect("Parse failed");

        assert_eq!(parsed.fields().len(), 0);
    }
}
