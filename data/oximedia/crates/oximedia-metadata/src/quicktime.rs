//! QuickTime user data atom parsing and writing support.
//!
//! QuickTime stores metadata in user data atoms within the 'udta' atom.
//!
//! # Format
//!
//! User data atoms have the format:
//! - Size (4 bytes, big-endian)
//! - Type (4 bytes, e.g., '©nam', '©ART', etc.)
//! - Data (variable length)
//!
//! # Common Atoms
//!
//! - **©nam**: Title
//! - **©ART**: Artist
//! - **©alb**: Album
//! - **©cmt**: Comment
//! - **©day**: Year
//! - **©cpy**: Copyright

use crate::{Error, Metadata, MetadataFormat, MetadataValue};
use std::io::{Cursor, Read};

/// Parse QuickTime user data from atom data.
///
/// # Errors
///
/// Returns an error if the data is not valid QuickTime user data.
pub fn parse(data: &[u8]) -> Result<Metadata, Error> {
    let mut metadata = Metadata::new(MetadataFormat::QuickTime);
    let mut cursor = Cursor::new(data);

    while cursor.position() < data.len() as u64 {
        // Read atom size
        let size = read_u32_be(&mut cursor)?;
        if size < 8 {
            break;
        }

        // Read atom type
        let mut type_bytes = [0u8; 4];
        cursor
            .read_exact(&mut type_bytes)
            .map_err(|e| Error::ParseError(format!("Failed to read atom type: {e}")))?;

        let atom_type = String::from_utf8_lossy(&type_bytes).to_string();

        // Read atom data
        let data_size = size.saturating_sub(8) as usize;
        let mut atom_data = vec![0u8; data_size];
        cursor
            .read_exact(&mut atom_data)
            .map_err(|e| Error::ParseError(format!("Failed to read atom data: {e}")))?;

        // Parse atom data
        let value = parse_atom_data(&atom_data)?;
        metadata.insert(atom_type, value);
    }

    Ok(metadata)
}

/// Parse atom data to metadata value.
fn parse_atom_data(data: &[u8]) -> Result<MetadataValue, Error> {
    // QuickTime user data is typically plain text
    // Remove null terminators
    let data_trimmed: Vec<u8> = data.iter().copied().filter(|&b| b != 0).collect();

    let text = String::from_utf8(data_trimmed)
        .map_err(|e| Error::EncodingError(format!("Invalid UTF-8 in QuickTime atom: {e}")))?;

    Ok(MetadataValue::Text(text))
}

/// Write QuickTime user data to atom data.
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn write(metadata: &Metadata) -> Result<Vec<u8>, Error> {
    let mut result = Vec::new();

    for (key, value) in metadata.fields() {
        if key.len() != 4 {
            continue; // Skip invalid atom types
        }

        if let Some(text) = value.as_text() {
            // Write atom size
            let size = (8 + text.len()) as u32;
            result.extend_from_slice(&size.to_be_bytes());

            // Write atom type
            result.extend_from_slice(key.as_bytes());

            // Write atom data
            result.extend_from_slice(text.as_bytes());
        }
    }

    Ok(result)
}

/// Read a 32-bit big-endian unsigned integer.
fn read_u32_be(cursor: &mut Cursor<&[u8]>) -> Result<u32, Error> {
    let mut bytes = [0u8; 4];
    cursor
        .read_exact(&mut bytes)
        .map_err(|e| Error::ParseError(format!("Failed to read u32: {e}")))?;
    Ok(u32::from_be_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quicktime_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::QuickTime);

        // Use ASCII atom names for testing
        metadata.insert(
            "test".to_string(),
            MetadataValue::Text("Test Title".to_string()),
        );
        metadata.insert(
            "artX".to_string(),
            MetadataValue::Text("Test Artist".to_string()),
        );
        metadata.insert(
            "albX".to_string(),
            MetadataValue::Text("Test Album".to_string()),
        );

        // Write
        let data = write(&metadata).expect("Write failed");

        // Parse
        let parsed = parse(&data).expect("Parse failed");

        assert_eq!(
            parsed.get("test").and_then(|v| v.as_text()),
            Some("Test Title")
        );
        assert_eq!(
            parsed.get("artX").and_then(|v| v.as_text()),
            Some("Test Artist")
        );
        assert_eq!(
            parsed.get("albX").and_then(|v| v.as_text()),
            Some("Test Album")
        );
    }

    #[test]
    fn test_quicktime_empty() {
        let metadata = Metadata::new(MetadataFormat::QuickTime);

        // Write
        let data = write(&metadata).expect("Write failed");

        // Should be empty
        assert_eq!(data.len(), 0);
    }
}
