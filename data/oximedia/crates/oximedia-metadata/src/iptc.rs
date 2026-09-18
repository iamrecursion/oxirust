//! IPTC (International Press Telecommunications Council) metadata parsing and writing support.
//!
//! IPTC-IIM (Information Interchange Model) is used for photo metadata.
//!
//! # Format
//!
//! IPTC data consists of datasets with the following structure:
//! - Tag marker (0x1C)
//! - Record number (1 byte)
//! - Dataset number (1 byte)
//! - Data length (2 bytes, big-endian)
//! - Data (variable length)
//!
//! # Common Datasets
//!
//! - **2:05**: Object Name (title)
//! - **2:80**: By-line (photographer/creator)
//! - **2:90**: City
//! - **2:101**: Country
//! - **2:116**: Copyright Notice
//! - **2:120**: Caption/Abstract

use crate::{Error, Metadata, MetadataFormat, MetadataValue};
use std::io::{Cursor, Read};

/// IPTC tag marker
const IPTC_TAG_MARKER: u8 = 0x1C;

/// IPTC record number for application data
const IPTC_RECORD_APP: u8 = 2;

/// Parse IPTC metadata from data.
///
/// # Errors
///
/// Returns an error if the data is not valid IPTC.
pub fn parse(data: &[u8]) -> Result<Metadata, Error> {
    let mut metadata = Metadata::new(MetadataFormat::Iptc);
    let mut cursor = Cursor::new(data);

    while (cursor.position() as usize) < data.len() {
        // Read tag marker
        let mut marker = [0u8; 1];
        if cursor.read_exact(&mut marker).is_err() {
            break;
        }

        if marker[0] != IPTC_TAG_MARKER {
            // Skip non-IPTC data
            continue;
        }

        // Read record number
        let mut record = [0u8; 1];
        if cursor.read_exact(&mut record).is_err() {
            break;
        }

        // Read dataset number
        let mut dataset = [0u8; 1];
        if cursor.read_exact(&mut dataset).is_err() {
            break;
        }

        // Read data length
        let length = read_u16_be(&mut cursor)?;

        // Read data
        let mut dataset_data = vec![0u8; length as usize];
        cursor
            .read_exact(&mut dataset_data)
            .map_err(|e| Error::ParseError(format!("Failed to read dataset data: {e}")))?;

        // Convert to text
        let text = String::from_utf8(dataset_data)
            .map_err(|e| Error::EncodingError(format!("Invalid UTF-8 in IPTC dataset: {e}")))?;

        // Get dataset name
        let dataset_name = get_dataset_name(record[0], dataset[0]);
        metadata.insert(dataset_name, MetadataValue::Text(text));
    }

    Ok(metadata)
}

/// Write IPTC metadata to data.
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn write(metadata: &Metadata) -> Result<Vec<u8>, Error> {
    let mut result = Vec::new();

    for (key, value) in metadata.fields() {
        if let Some(text) = value.as_text() {
            let (record, dataset) = get_dataset_id(key);

            // Write tag marker
            result.push(IPTC_TAG_MARKER);

            // Write record number
            result.push(record);

            // Write dataset number
            result.push(dataset);

            // Write data length
            let length = text.len() as u16;
            result.extend_from_slice(&length.to_be_bytes());

            // Write data
            result.extend_from_slice(text.as_bytes());
        }
    }

    Ok(result)
}

/// Get dataset name from record and dataset numbers.
fn get_dataset_name(record: u8, dataset: u8) -> String {
    if record == IPTC_RECORD_APP {
        match dataset {
            5 => "ObjectName".to_string(),
            80 => "By-line".to_string(),
            90 => "City".to_string(),
            95 => "Province-State".to_string(),
            101 => "Country-PrimaryLocationName".to_string(),
            116 => "CopyrightNotice".to_string(),
            120 => "Caption-Abstract".to_string(),
            122 => "Writer-Editor".to_string(),
            25 => "Keywords".to_string(),
            55 => "DateCreated".to_string(),
            _ => format!("Dataset_{}:{}", record, dataset),
        }
    } else {
        format!("Dataset_{}:{}", record, dataset)
    }
}

/// Get record and dataset numbers from dataset name.
fn get_dataset_id(name: &str) -> (u8, u8) {
    match name {
        "ObjectName" => (IPTC_RECORD_APP, 5),
        "By-line" => (IPTC_RECORD_APP, 80),
        "City" => (IPTC_RECORD_APP, 90),
        "Province-State" => (IPTC_RECORD_APP, 95),
        "Country-PrimaryLocationName" => (IPTC_RECORD_APP, 101),
        "CopyrightNotice" => (IPTC_RECORD_APP, 116),
        "Caption-Abstract" => (IPTC_RECORD_APP, 120),
        "Writer-Editor" => (IPTC_RECORD_APP, 122),
        "Keywords" => (IPTC_RECORD_APP, 25),
        "DateCreated" => (IPTC_RECORD_APP, 55),
        _ => (IPTC_RECORD_APP, 0),
    }
}

/// Read a 16-bit big-endian unsigned integer.
fn read_u16_be(cursor: &mut Cursor<&[u8]>) -> Result<u16, Error> {
    let mut bytes = [0u8; 2];
    cursor
        .read_exact(&mut bytes)
        .map_err(|e| Error::ParseError(format!("Failed to read u16: {e}")))?;
    Ok(u16::from_be_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iptc_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::Iptc);

        metadata.insert(
            "ObjectName".to_string(),
            MetadataValue::Text("Test Title".to_string()),
        );
        metadata.insert(
            "By-line".to_string(),
            MetadataValue::Text("Test Photographer".to_string()),
        );
        metadata.insert(
            "CopyrightNotice".to_string(),
            MetadataValue::Text("Copyright 2024".to_string()),
        );

        // Write
        let data = write(&metadata).expect("Write failed");

        // Parse
        let parsed = parse(&data).expect("Parse failed");

        assert_eq!(
            parsed.get("ObjectName").and_then(|v| v.as_text()),
            Some("Test Title")
        );
        assert_eq!(
            parsed.get("By-line").and_then(|v| v.as_text()),
            Some("Test Photographer")
        );
        assert_eq!(
            parsed.get("CopyrightNotice").and_then(|v| v.as_text()),
            Some("Copyright 2024")
        );
    }

    #[test]
    fn test_get_dataset_name() {
        assert_eq!(get_dataset_name(2, 5), "ObjectName");
        assert_eq!(get_dataset_name(2, 80), "By-line");
        assert_eq!(get_dataset_name(2, 116), "CopyrightNotice");
        assert_eq!(get_dataset_name(2, 120), "Caption-Abstract");
    }

    #[test]
    fn test_get_dataset_id() {
        assert_eq!(get_dataset_id("ObjectName"), (2, 5));
        assert_eq!(get_dataset_id("By-line"), (2, 80));
        assert_eq!(get_dataset_id("CopyrightNotice"), (2, 116));
        assert_eq!(get_dataset_id("Caption-Abstract"), (2, 120));
    }
}
