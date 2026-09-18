//! Scope snapshot — export scope displays as PNG-compatible image data
//! with metadata annotations.
//!
//! This module provides a pure-Rust PNG encoder (without external image
//! crates) that can serialise any RGBA scope image to a valid PNG byte stream.
//! Metadata is embedded as PNG tEXt chunks.
//!
//! # Features
//!
//! - Minimal deflate-based PNG encoder (IDAT with zlib wrapping, Adler-32)
//! - PNG tEXt metadata chunks (keyword/value pairs, up to 79-byte keywords)
//! - `ScopeSnapshotConfig` for output path, metadata, and compression hints
//! - `ScopeSnapshot` struct that captures RGBA data + metadata at an instant
//! - Deterministic output (same input → same bytes)
//!
//! # PNG Encoding Notes
//!
//! The encoder uses no compression (deflate level 0) to remain dependency-free
//! while producing valid, universally decodable PNG files.  The output is
//! larger than a fully compressed PNG but is correct per the PNG specification.

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use oximedia_core::{OxiError, OxiResult};

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot metadata
// ─────────────────────────────────────────────────────────────────────────────

/// A key/value metadata pair to embed in the PNG tEXt chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataEntry {
    /// PNG tEXt keyword (max 79 bytes, Latin-1, no null bytes).
    pub key: String,
    /// Metadata value (arbitrary text).
    pub value: String,
}

impl MetadataEntry {
    /// Creates a new entry, truncating the key to 79 bytes if necessary.
    #[must_use]
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        let mut k = key.into();
        k.truncate(79);
        Self {
            key: k,
            value: value.into(),
        }
    }
}

/// Configuration for `export_scope_snapshot`.
#[derive(Debug, Clone)]
pub struct ScopeSnapshotConfig {
    /// Metadata entries to embed as PNG tEXt chunks.
    pub metadata: Vec<MetadataEntry>,
    /// Whether to include a standard "Software" tEXt entry.
    pub include_software_tag: bool,
}

impl Default for ScopeSnapshotConfig {
    fn default() -> Self {
        Self {
            metadata: Vec::new(),
            include_software_tag: true,
        }
    }
}

/// A captured scope image with associated metadata.
#[derive(Debug, Clone)]
pub struct ScopeSnapshot {
    /// RGBA pixel data (width × height × 4 bytes).
    pub rgba: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Metadata associated with this snapshot.
    pub metadata: Vec<MetadataEntry>,
}

impl ScopeSnapshot {
    /// Creates a new `ScopeSnapshot`.
    ///
    /// # Errors
    ///
    /// Returns an error if `rgba.len() != width * height * 4`.
    pub fn new(
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        metadata: Vec<MetadataEntry>,
    ) -> OxiResult<Self> {
        let expected = (width * height * 4) as usize;
        if rgba.len() != expected {
            return Err(OxiError::InvalidData(format!(
                "RGBA length {} != expected {expected}",
                rgba.len()
            )));
        }
        Ok(Self {
            rgba,
            width,
            height,
            metadata,
        })
    }

    /// Encode this snapshot to a valid PNG byte stream.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails (should not happen for valid data).
    pub fn encode_png(&self, config: &ScopeSnapshotConfig) -> OxiResult<Vec<u8>> {
        export_scope_snapshot(&self.rgba, self.width, self.height, config)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PNG encoder
// ─────────────────────────────────────────────────────────────────────────────

/// Encode an RGBA image as a minimal but valid PNG byte stream.
///
/// # Arguments
///
/// * `rgba` — RGBA pixel data, length must be `width * height * 4`.
/// * `width` / `height` — image dimensions.
/// * `config` — metadata and encoding options.
///
/// # Errors
///
/// Returns an error if the buffer length does not match the declared dimensions.
pub fn export_scope_snapshot(
    rgba: &[u8],
    width: u32,
    height: u32,
    config: &ScopeSnapshotConfig,
) -> OxiResult<Vec<u8>> {
    let expected = (width * height * 4) as usize;
    if rgba.len() != expected {
        return Err(OxiError::InvalidData(format!(
            "RGBA length {} != expected {expected}",
            rgba.len()
        )));
    }

    let mut out: Vec<u8> = Vec::with_capacity(expected + 256);

    // PNG signature
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    // IHDR chunk
    write_ihdr(&mut out, width, height);

    // tEXt metadata chunks
    if config.include_software_tag {
        write_text_chunk(&mut out, "Software", "oximedia-scopes");
    }
    for entry in &config.metadata {
        write_text_chunk(&mut out, &entry.key, &entry.value);
    }

    // IDAT chunk (uncompressed zlib)
    write_idat(&mut out, rgba, width, height);

    // IEND chunk
    write_iend(&mut out);

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// PNG chunk helpers
// ─────────────────────────────────────────────────────────────────────────────

fn write_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    let len = data.len() as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let crc = crc32_ieee(chunk_type, data);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn write_ihdr(out: &mut Vec<u8>, width: u32, height: u32) {
    let mut data = [0u8; 13];
    data[0..4].copy_from_slice(&width.to_be_bytes());
    data[4..8].copy_from_slice(&height.to_be_bytes());
    data[8] = 8; // bit depth
    data[9] = 6; // colour type: RGBA
    data[10] = 0; // compression method
    data[11] = 0; // filter method
    data[12] = 0; // interlace method
    write_chunk(out, b"IHDR", &data);
}

fn write_text_chunk(out: &mut Vec<u8>, keyword: &str, value: &str) {
    let mut data = Vec::with_capacity(keyword.len() + 1 + value.len());
    data.extend_from_slice(keyword.as_bytes());
    data.push(0); // null separator
    data.extend_from_slice(value.as_bytes());
    write_chunk(out, b"tEXt", &data);
}

fn write_iend(out: &mut Vec<u8>) {
    write_chunk(out, b"IEND", &[]);
}

/// Write an uncompressed zlib IDAT stream with PNG filter byte 0 (None).
fn write_idat(out: &mut Vec<u8>, rgba: &[u8], width: u32, height: u32) {
    let w = width as usize;
    let h = height as usize;
    let row_bytes = w * 4; // RGBA

    // Build raw filtered scanlines (filter byte 0 = None prepended to each row)
    let raw_len = h * (1 + row_bytes);
    let mut raw = Vec::with_capacity(raw_len);
    for row in 0..h {
        raw.push(0u8); // filter type None
        raw.extend_from_slice(&rgba[row * row_bytes..(row + 1) * row_bytes]);
    }

    // Wrap in a zlib stream (deflate stored block, no compression)
    let zlib = zlib_store(&raw);
    write_chunk(out, b"IDAT", &zlib);
}

/// Produce a zlib-wrapped stored (no compression) deflate stream.
fn zlib_store(data: &[u8]) -> Vec<u8> {
    // zlib header: CMF=0x78 (deflate, window=32K), FLG=0x01 (no dict, fcheck)
    // 0x7801 mod 31 == 0
    let mut out = Vec::with_capacity(data.len() + 6);
    out.push(0x78);
    out.push(0x01);

    // Deflate BFINAL/BTYPE=00 (stored), split into 65535-byte blocks
    let chunk_size = 65535usize;
    let mut pos = 0;
    while pos < data.len() || data.is_empty() {
        let end = (pos + chunk_size).min(data.len());
        let last = end >= data.len();
        let block = &data[pos..end];
        let blen = block.len() as u16;
        let nlen = !blen;
        // BFINAL | BTYPE (00 = stored)
        out.push(if last { 0x01 } else { 0x00 });
        out.extend_from_slice(&blen.to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(block);
        pos = end;
        if data.is_empty() {
            break;
        }
    }

    // Adler-32 checksum
    let adler = adler32(data);
    out.extend_from_slice(&adler.to_be_bytes());
    out
}

/// Adler-32 checksum (RFC 1950).
fn adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65521;
    let mut a = 1u32;
    let mut b = 0u32;
    for &byte in data {
        a = (a + u32::from(byte)) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    (b << 16) | a
}

/// CRC-32 (IEEE 802.3 polynomial) for PNG chunk validation.
fn crc32_ieee(chunk_type: &[u8], data: &[u8]) -> u32 {
    // CRC table (pre-computed would be faster but this is inlined for clarity)
    fn make_table() -> [u32; 256] {
        let mut table = [0u32; 256];
        for n in 0..256u32 {
            let mut c = n;
            for _ in 0..8 {
                if c & 1 != 0 {
                    c = 0xEDB8_8320 ^ (c >> 1);
                } else {
                    c >>= 1;
                }
            }
            table[n as usize] = c;
        }
        table
    }
    let table = make_table();
    let mut crc = !0u32;
    for &b in chunk_type.iter().chain(data.iter()) {
        crc = table[((crc ^ u32::from(b)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.extend_from_slice(&color);
        }
        v
    }

    #[test]
    fn test_metadata_entry_new() {
        let e = MetadataEntry::new("Author", "Test Suite");
        assert_eq!(e.key, "Author");
        assert_eq!(e.value, "Test Suite");
    }

    #[test]
    fn test_metadata_entry_key_truncated() {
        let long_key = "A".repeat(200);
        let e = MetadataEntry::new(long_key, "v");
        assert_eq!(e.key.len(), 79);
    }

    #[test]
    fn test_scope_snapshot_new_valid() {
        let rgba = solid_rgba(4, 4, [255, 0, 0, 255]);
        let snap = ScopeSnapshot::new(rgba, 4, 4, vec![]);
        assert!(snap.is_ok());
    }

    #[test]
    fn test_scope_snapshot_new_invalid_size() {
        let rgba = vec![0u8; 10];
        let snap = ScopeSnapshot::new(rgba, 4, 4, vec![]);
        assert!(snap.is_err());
    }

    #[test]
    fn test_export_scope_snapshot_wrong_size() {
        let rgba = vec![0u8; 10];
        let cfg = ScopeSnapshotConfig::default();
        let result = export_scope_snapshot(&rgba, 4, 4, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_export_scope_snapshot_produces_png_signature() {
        let rgba = solid_rgba(2, 2, [128, 128, 128, 255]);
        let cfg = ScopeSnapshotConfig::default();
        let result = export_scope_snapshot(&rgba, 2, 2, &cfg);
        assert!(result.is_ok());
        let png = result.expect("should succeed");
        // PNG magic bytes
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn test_export_scope_snapshot_contains_ihdr() {
        let rgba = solid_rgba(4, 4, [0, 0, 255, 255]);
        let cfg = ScopeSnapshotConfig::default();
        let png = export_scope_snapshot(&rgba, 4, 4, &cfg).expect("should succeed");
        // IHDR starts at byte 8 (after signature), length field then type
        let ihdr_type = &png[12..16];
        assert_eq!(ihdr_type, b"IHDR");
    }

    #[test]
    fn test_export_scope_snapshot_iend_at_end() {
        let rgba = solid_rgba(4, 4, [0, 255, 0, 255]);
        let cfg = ScopeSnapshotConfig::default();
        let png = export_scope_snapshot(&rgba, 4, 4, &cfg).expect("should succeed");
        // Last 12 bytes: 4-byte length (0), 4-byte "IEND", 4-byte CRC
        let iend_type = &png[png.len() - 8..png.len() - 4];
        assert_eq!(iend_type, b"IEND");
    }

    #[test]
    fn test_export_scope_snapshot_with_metadata() {
        let rgba = solid_rgba(4, 4, [255, 255, 0, 255]);
        let cfg = ScopeSnapshotConfig {
            metadata: vec![
                MetadataEntry::new("Scope", "Waveform"),
                MetadataEntry::new("Frame", "42"),
            ],
            include_software_tag: true,
        };
        let result = export_scope_snapshot(&rgba, 4, 4, &cfg);
        assert!(result.is_ok());
        let png = result.expect("should succeed");
        // The PNG should contain the keyword "Scope"
        let png_str = String::from_utf8_lossy(&png);
        assert!(png_str.contains("Scope"));
    }

    #[test]
    fn test_export_scope_snapshot_no_software_tag() {
        let rgba = solid_rgba(2, 2, [10, 20, 30, 255]);
        let cfg = ScopeSnapshotConfig {
            include_software_tag: false,
            metadata: vec![],
        };
        let result = export_scope_snapshot(&rgba, 2, 2, &cfg);
        assert!(result.is_ok());
        let png = result.expect("should succeed");
        assert!(!String::from_utf8_lossy(&png).contains("Software"));
    }

    #[test]
    fn test_scope_snapshot_encode_png() {
        let rgba = solid_rgba(8, 8, [200, 100, 50, 255]);
        let snap = ScopeSnapshot::new(rgba, 8, 8, vec![MetadataEntry::new("Test", "ok")])
            .expect("valid snap");
        let cfg = ScopeSnapshotConfig::default();
        let png = snap.encode_png(&cfg).expect("encode png");
        assert_eq!(&png[0..4], &[137, 80, 78, 71]);
    }

    #[test]
    fn test_adler32_empty() {
        assert_eq!(adler32(&[]), 1); // Adler-32 of empty = 1
    }

    #[test]
    fn test_adler32_known() {
        // "Wikipedia" example: adler32("Wikipedia") = 0x11E60398
        let result = adler32(b"Wikipedia");
        assert_eq!(result, 0x11E6_0398);
    }

    #[test]
    fn test_png_deterministic_output() {
        let rgba = solid_rgba(4, 4, [1, 2, 3, 255]);
        let cfg = ScopeSnapshotConfig::default();
        let png1 = export_scope_snapshot(&rgba, 4, 4, &cfg).expect("encode 1");
        let png2 = export_scope_snapshot(&rgba, 4, 4, &cfg).expect("encode 2");
        assert_eq!(png1, png2);
    }
}
