//! AAF low-level dump/inspection tool
//!
//! Provides hex + structure view for debugging corrupt or unknown AAF files.
//! Supports byte-level hex dump with ASCII side-by-side, structured storage
//! directory listing, and field-level annotations for AAF binary structures.

use crate::structured_storage::{ObjectType, StorageReader};
use std::fmt::Write as FmtWrite;
use std::io::{Read, Seek};

/// Width (columns) of each hex dump row
const HEX_COLS: usize = 16;

/// Render a byte slice as a hex dump with offsets and ASCII column.
///
/// # Example output
///
/// ```text
/// 00000000  d0 cf 11 e0 a1 b1 1a e1  00 00 00 00 00 00 00 00  |................|
/// 00000010  3e 00 03 00 fe ff 09 00  06 00 00 00 00 00 00 00  |>...............|
/// ```
#[must_use]
pub fn hex_dump(data: &[u8], base_offset: usize) -> String {
    let mut out = String::new();

    for (chunk_idx, chunk) in data.chunks(HEX_COLS).enumerate() {
        let offset = base_offset + chunk_idx * HEX_COLS;

        // Offset column
        let _ = write!(out, "{offset:08x}  ");

        // Hex bytes (two groups of 8)
        for (i, b) in chunk.iter().enumerate() {
            if i == 8 {
                out.push(' ');
            }
            let _ = write!(out, "{b:02x} ");
        }

        // Pad short rows
        let pad_cols = HEX_COLS - chunk.len();
        for i in 0..pad_cols {
            if chunk.len() + i == 8 {
                out.push(' ');
            }
            out.push_str("   ");
        }

        // ASCII column
        out.push_str(" |");
        for &b in chunk {
            if b.is_ascii_graphic() || b == b' ' {
                out.push(b as char);
            } else {
                out.push('.');
            }
        }
        out.push_str("|\n");
    }

    out
}

/// Parse and annotate the structured storage header bytes.
///
/// Returns a multi-line string describing each header field and its value.
#[must_use]
pub fn dump_cfb_header(data: &[u8]) -> String {
    let mut out = String::new();

    if data.len() < 512 {
        let _ = writeln!(
            out,
            "ERROR: data too short for CFB header ({} bytes)",
            data.len()
        );
        return out;
    }

    // Signature
    let _ = writeln!(out, "--- Compound File Binary Header ---");
    let sig = &data[0..8];
    let sig_ok = sig == b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1";
    let _ = writeln!(
        out,
        "  [0x00-0x07] Signature:         {:?} {}",
        sig,
        if sig_ok { "(valid)" } else { "(INVALID!)" }
    );

    let clsid = &data[8..24];
    let _ = writeln!(out, "  [0x08-0x17] CLSID:             {:?}", clsid);

    let minor_ver = u16::from_le_bytes([data[24], data[25]]);
    let major_ver = u16::from_le_bytes([data[26], data[27]]);
    let _ = writeln!(out, "  [0x18-0x19] Minor version:     0x{minor_ver:04x}");
    let _ = writeln!(
        out,
        "  [0x1a-0x1b] Major version:     {} {}",
        major_ver,
        if major_ver == 3 || major_ver == 4 {
            "(supported)"
        } else {
            "(UNSUPPORTED)"
        }
    );

    let byte_order = u16::from_le_bytes([data[28], data[29]]);
    let _ = writeln!(
        out,
        "  [0x1c-0x1d] Byte order:        0x{byte_order:04x} {}",
        if byte_order == 0xFFFE {
            "(little-endian)"
        } else {
            "(non-standard)"
        }
    );

    let sector_shift = u16::from_le_bytes([data[30], data[31]]);
    let sector_size = 1u64 << sector_shift;
    let _ = writeln!(
        out,
        "  [0x1e-0x1f] Sector shift:      {sector_shift} -> sector size = {sector_size} bytes"
    );

    let mini_shift = u16::from_le_bytes([data[32], data[33]]);
    let mini_size = 1u64 << mini_shift;
    let _ = writeln!(
        out,
        "  [0x20-0x21] Mini sector shift: {mini_shift} -> mini sector size = {mini_size} bytes"
    );

    let total_sectors = u32::from_le_bytes([data[40], data[41], data[42], data[43]]);
    let _ = writeln!(out, "  [0x28-0x2b] Total sectors:     {total_sectors}");

    let fat_sectors = u32::from_le_bytes([data[44], data[45], data[46], data[47]]);
    let _ = writeln!(out, "  [0x2c-0x2f] FAT sectors:       {fat_sectors}");

    let first_dir = u32::from_le_bytes([data[48], data[49], data[50], data[51]]);
    let _ = writeln!(out, "  [0x30-0x33] First dir sector:  {first_dir}");

    out
}

/// A single line in a structured storage directory listing.
#[derive(Debug, Clone)]
pub struct DirectoryListing {
    /// Directory entry index
    pub index: usize,
    /// Entry name
    pub name: String,
    /// Object type string
    pub object_type: &'static str,
    /// Size in bytes
    pub size: u64,
    /// Starting sector
    pub starting_sector: u32,
}

impl DirectoryListing {
    /// Format as a table row
    #[must_use]
    pub fn to_row(&self) -> String {
        format!(
            "{:>4}  {:<32}  {:<10}  {:>12}  {:>10}",
            self.index, self.name, self.object_type, self.size, self.starting_sector
        )
    }
}

/// List all directory entries in a structured storage file.
pub fn list_directory<R: Read + Seek>(reader: &mut StorageReader<R>) -> Vec<DirectoryListing> {
    reader
        .directory_entries()
        .iter()
        .enumerate()
        .map(|(idx, entry)| {
            let object_type = match entry.object_type {
                ObjectType::Root => "Root",
                ObjectType::Storage => "Storage",
                ObjectType::Stream => "Stream",
                ObjectType::Unknown => "Unknown",
            };
            DirectoryListing {
                index: idx,
                name: entry.name.clone(),
                object_type,
                size: entry.size,
                starting_sector: entry.starting_sector,
            }
        })
        .collect()
}

/// Render the directory listing as an ASCII table.
#[must_use]
pub fn render_directory_table(listings: &[DirectoryListing]) -> String {
    let mut out = String::new();
    let header = format!(
        "{:>4}  {:<32}  {:<10}  {:>12}  {:>10}",
        "IDX", "NAME", "TYPE", "SIZE (bytes)", "SECTOR"
    );
    let _ = writeln!(out, "{header}");
    let _ = writeln!(out, "{}", "-".repeat(header.len()));
    for listing in listings {
        let _ = writeln!(out, "{}", listing.to_row());
    }
    out
}

/// Full inspection report for a structured storage stream.
#[derive(Debug)]
pub struct InspectionReport {
    /// Header annotation
    pub header_annotation: String,
    /// Directory listing
    pub directory: Vec<DirectoryListing>,
    /// Whether the CFB signature is valid
    pub signature_valid: bool,
    /// Detected major version
    pub major_version: Option<u16>,
    /// Total number of directory entries
    pub entry_count: usize,
}

/// Produce a full inspection report from raw CFB bytes and an open reader.
///
/// `raw_header_bytes` must be at least the first 512 bytes of the file.
pub fn inspect<R: Read + Seek>(
    raw_header_bytes: &[u8],
    reader: &mut StorageReader<R>,
) -> InspectionReport {
    let header_annotation = dump_cfb_header(raw_header_bytes);

    let signature_valid = raw_header_bytes.len() >= 8
        && &raw_header_bytes[0..8] == b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1";

    let major_version = if raw_header_bytes.len() >= 28 {
        Some(u16::from_le_bytes([
            raw_header_bytes[26],
            raw_header_bytes[27],
        ]))
    } else {
        None
    };

    let directory = list_directory(reader);
    let entry_count = directory.len();

    InspectionReport {
        header_annotation,
        directory,
        signature_valid,
        major_version,
        entry_count,
    }
}

/// Render the full inspection report as a human-readable string.
#[must_use]
pub fn render_report(report: &InspectionReport) -> String {
    let mut out = report.header_annotation.clone();
    let _ = writeln!(out, "\n--- Directory Entries ({}) ---", report.entry_count);
    out.push_str(&render_directory_table(&report.directory));
    out
}

/// Dump a slice of bytes as annotated hex for debugging.
///
/// Useful for examining raw property or essence stream bytes.
#[must_use]
pub fn dump_bytes_annotated(data: &[u8], label: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "=== {label} ({} bytes) ===", data.len());
    out.push_str(&hex_dump(data, 0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_dump_empty() {
        let dump = hex_dump(&[], 0);
        assert!(dump.is_empty());
    }

    #[test]
    fn test_hex_dump_single_byte() {
        let dump = hex_dump(&[0xAB], 0);
        assert!(dump.contains("ab"));
        assert!(dump.contains("|.|"));
    }

    #[test]
    fn test_hex_dump_printable_ascii() {
        let dump = hex_dump(b"Hello", 0);
        assert!(dump.contains("48 65 6c 6c 6f"));
        assert!(dump.contains("|Hello|"));
    }

    #[test]
    fn test_hex_dump_two_rows() {
        let data = [0u8; 32];
        let dump = hex_dump(&data, 0);
        // Should have two rows: offset 00000000 and 00000010
        assert!(dump.contains("00000000"));
        assert!(dump.contains("00000010"));
    }

    #[test]
    fn test_hex_dump_base_offset() {
        let dump = hex_dump(&[0x00], 0x100);
        assert!(dump.contains("00000100"));
    }

    #[test]
    fn test_dump_cfb_header_too_short() {
        let result = dump_cfb_header(&[0u8; 4]);
        assert!(result.contains("ERROR"));
    }

    #[test]
    fn test_dump_cfb_header_valid_signature() {
        let mut data = vec![0u8; 512];
        // Write CFB signature
        data[0..8].copy_from_slice(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1");
        let result = dump_cfb_header(&data);
        assert!(result.contains("valid)"));
    }

    #[test]
    fn test_dump_cfb_header_invalid_signature() {
        let data = vec![0u8; 512];
        let result = dump_cfb_header(&data);
        assert!(result.contains("INVALID"));
    }

    #[test]
    fn test_directory_listing_to_row() {
        let listing = DirectoryListing {
            index: 0,
            name: "Root Entry".to_string(),
            object_type: "Root",
            size: 0,
            starting_sector: 0xFFFFFFFE,
        };
        let row = listing.to_row();
        assert!(row.contains("Root Entry"));
        assert!(row.contains("Root"));
    }

    #[test]
    fn test_render_directory_table_empty() {
        let table = render_directory_table(&[]);
        // Header line + separator line
        assert!(table.contains("NAME"));
        assert!(table.contains("TYPE"));
    }

    #[test]
    fn test_dump_bytes_annotated() {
        let result = dump_bytes_annotated(b"AAF", "Test");
        assert!(result.contains("Test"));
        assert!(result.contains("3 bytes"));
        assert!(result.contains("41 41 46"));
    }
}
