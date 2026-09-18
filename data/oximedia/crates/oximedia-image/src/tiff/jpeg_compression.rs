//! JPEG-in-TIFF compression (TIFF compression 7, "new-style" JPEG).
//!
//! TIFF 6.0 Technical Note 2 specifies JPEG compression where each strip (or
//! tile) carries a baseline JPEG datastream. Two layouts exist:
//!
//! 1. **Self-contained** — each strip is a complete JFIF stream beginning with
//!    SOI and ending with EOI, carrying its own DQT/DHT tables.
//! 2. **Abbreviated** — the strips carry only the entropy-coded scan; the
//!    shared DQT/DHT tables live in the `JPEGTables` (tag 347) IFD entry as an
//!    abbreviated JPEG datastream (SOI ... tables ... EOI).
//!
//! The decoder here accepts both. When `JPEGTables` is present and a strip is
//! abbreviated, the table segments are spliced in front of the strip's scan
//! before handing the result to the baseline JPEG decoder.
//!
//! The encoder always produces complete self-contained JFIF strips (the
//! simplest interoperable layout) and *also* emits a `JPEGTables` entry so a
//! strict TIFF reader sees the tag — the strips remain independently decodable.

#![allow(clippy::cast_possible_truncation)]

use crate::error::{ImageError, ImageResult};
use crate::jpeg::{JpegDecoder, JpegEncoder, JpegQuality, JPEG_DHT, JPEG_DQT, JPEG_EOI, JPEG_SOI};
use crate::{ColorSpace, ImageData, ImageFrame, PixelType};

/// JPEG marker prefix byte.
const MARKER_PREFIX: u8 = 0xFF;
/// SOS (Start of Scan) marker low byte.
const SOS_LOW: u8 = 0xDA;
/// SOI low byte.
const SOI_LOW: u8 = 0xD8;
/// EOI low byte.
const EOI_LOW: u8 = 0xD9;

/// Decode a single JPEG-compressed TIFF strip.
///
/// * `strip` — the raw strip bytes from the file.
/// * `jpeg_tables` — contents of the `JPEGTables` (347) tag, if present.
///
/// Returns interleaved 8-bit pixel data (grey or RGB) for the strip.
pub fn decode_jpeg_strip(strip: &[u8], jpeg_tables: Option<&[u8]>) -> ImageResult<Vec<u8>> {
    let full = assemble_jpeg_stream(strip, jpeg_tables)?;
    let frame = JpegDecoder::new().decode(&full)?;
    Ok(frame.pixels)
}

/// Build a complete, decodable JFIF stream from a TIFF strip plus optional
/// shared `JPEGTables`.
fn assemble_jpeg_stream(strip: &[u8], jpeg_tables: Option<&[u8]>) -> ImageResult<Vec<u8>> {
    if strip.len() < 2 {
        return Err(ImageError::invalid_format("JPEG-in-TIFF: strip too short"));
    }

    let strip_has_soi = strip[0] == MARKER_PREFIX && strip[1] == SOI_LOW;

    // Case 1: strip already a complete JFIF stream.
    if strip_has_soi {
        // A self-contained strip already carries every table. If JPEGTables is
        // also present we splice its DQT/DHT segments after the SOI; duplicate
        // tables are idempotent for the baseline decoder.
        if let Some(tables) = jpeg_tables {
            let table_segs = extract_table_segments(tables);
            if table_segs.is_empty() {
                return Ok(strip.to_vec());
            }
            let mut out = Vec::with_capacity(strip.len() + table_segs.len());
            out.extend_from_slice(&JPEG_SOI.to_be_bytes());
            out.extend_from_slice(&table_segs);
            out.extend_from_slice(&strip[2..]);
            return Ok(out);
        }
        return Ok(strip.to_vec());
    }

    // Case 2: abbreviated strip — needs the shared tables prepended.
    let tables = jpeg_tables.ok_or_else(|| {
        ImageError::invalid_format("JPEG-in-TIFF: abbreviated strip without JPEGTables tag")
    })?;
    let table_segs = extract_table_segments(tables);

    let mut out = Vec::with_capacity(table_segs.len() + strip.len() + 4);
    out.extend_from_slice(&JPEG_SOI.to_be_bytes());
    out.extend_from_slice(&table_segs);
    out.extend_from_slice(strip);
    if !ends_with_eoi(&out) {
        out.extend_from_slice(&JPEG_EOI.to_be_bytes());
    }
    Ok(out)
}

/// Returns `true` if `data` ends with an EOI marker.
fn ends_with_eoi(data: &[u8]) -> bool {
    data.len() >= 2 && data[data.len() - 2] == MARKER_PREFIX && data[data.len() - 1] == EOI_LOW
}

/// Extract the DQT and DHT marker segments from an abbreviated JPEG table
/// stream (the `JPEGTables` tag payload). SOI/EOI and any SOS are dropped.
fn extract_table_segments(tables: &[u8]) -> Vec<u8> {
    let dqt_low = (JPEG_DQT & 0xFF) as u8;
    let dht_low = (JPEG_DHT & 0xFF) as u8;
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 1 < tables.len() {
        if tables[i] != MARKER_PREFIX {
            i += 1;
            continue;
        }
        let marker = tables[i + 1];
        match marker {
            // SOI / EOI / TEM / padding: standalone, no length field.
            SOI_LOW | EOI_LOW | 0x01 | MARKER_PREFIX => {
                i += 2;
                continue;
            }
            // RST0..RST7 are standalone too.
            0xD0..=0xD7 => {
                i += 2;
                continue;
            }
            SOS_LOW => break, // tables never contain a scan
            _ => {}
        }
        // Length-bearing segment: 2 marker bytes + 2 length bytes + payload.
        if i + 3 >= tables.len() {
            break;
        }
        let seg_len = u16::from_be_bytes([tables[i + 2], tables[i + 3]]) as usize;
        if seg_len < 2 || i + 2 + seg_len > tables.len() {
            break;
        }
        let seg_end = i + 2 + seg_len;
        if marker == dqt_low || marker == dht_low {
            out.extend_from_slice(&tables[i..seg_end]);
        }
        i = seg_end;
    }
    out
}

/// Encode interleaved 8-bit pixel data as a JPEG-compressed TIFF strip.
///
/// Produces a complete, self-contained JFIF datastream so the strip is
/// independently decodable regardless of the `JPEGTables` tag.
pub fn encode_jpeg_strip(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    quality: u8,
) -> ImageResult<Vec<u8>> {
    let color_space = if components == 1 {
        ColorSpace::Luma
    } else {
        ColorSpace::Srgb
    };
    let frame = ImageFrame::new(
        0,
        width,
        height,
        PixelType::U8,
        components,
        color_space,
        ImageData::interleaved(pixels.to_vec()),
    );
    let encoder = JpegEncoder::new(JpegQuality::new(quality));
    encoder.encode(&frame)
}

/// Build the `JPEGTables` tag payload: an abbreviated JPEG datastream
/// (SOI, the DQT/DHT segments produced for the given quality, EOI).
///
/// This mirrors the tables `encode_jpeg_strip` writes into each strip, so a
/// reader may rely on either source.
pub fn build_jpeg_tables(components: u8, quality: u8) -> ImageResult<Vec<u8>> {
    // Encode a tiny dummy frame and harvest its table segments — this keeps the
    // tables byte-identical to those embedded in real strips.
    let dummy = encode_jpeg_strip(&vec![128u8; components as usize], 1, 1, components, quality)?;
    let segs = extract_table_segments(&dummy);
    let mut out = Vec::with_capacity(segs.len() + 4);
    out.extend_from_slice(&JPEG_SOI.to_be_bytes());
    out.extend_from_slice(&segs);
    out.extend_from_slice(&JPEG_EOI.to_be_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_grey_strip() {
        let width = 16u32;
        let height = 16u32;
        let pixels: Vec<u8> = (0..(width * height)).map(|i| (i % 256) as u8).collect();
        let strip = encode_jpeg_strip(&pixels, width, height, 1, 90).expect("encode");
        let decoded = decode_jpeg_strip(&strip, None).expect("decode");
        assert_eq!(decoded.len(), pixels.len());
    }

    #[test]
    fn test_roundtrip_rgb_strip() {
        let width = 16u32;
        let height = 16u32;
        let mut pixels = Vec::new();
        for y in 0..height {
            for x in 0..width {
                pixels.push((x * 16) as u8);
                pixels.push((y * 16) as u8);
                pixels.push(128u8);
            }
        }
        let strip = encode_jpeg_strip(&pixels, width, height, 3, 90).expect("encode");
        let decoded = decode_jpeg_strip(&strip, None).expect("decode");
        assert_eq!(decoded.len(), pixels.len());
    }

    #[test]
    fn test_build_jpeg_tables_grey() {
        let tables = build_jpeg_tables(1, 75).expect("tables");
        assert_eq!(&tables[0..2], &JPEG_SOI.to_be_bytes());
        assert_eq!(&tables[tables.len() - 2..], &JPEG_EOI.to_be_bytes());
        let segs = extract_table_segments(&tables);
        assert!(!segs.is_empty(), "tables must carry DQT/DHT segments");
    }

    #[test]
    fn test_extract_segments_drops_soi_eoi() {
        let tables = build_jpeg_tables(3, 80).expect("tables");
        let segs = extract_table_segments(&tables);
        assert!(segs.len() >= 4);
        assert!(!(segs[0] == 0xFF && segs[1] == SOI_LOW));
    }

    #[test]
    fn test_abbreviated_strip_needs_tables() {
        let fake_scan = vec![0x12u8, 0x34, 0x56];
        let err = decode_jpeg_strip(&fake_scan, None);
        assert!(err.is_err());
    }

    #[test]
    fn test_self_contained_strip_with_extra_tables() {
        let width = 8u32;
        let height = 8u32;
        let pixels = vec![100u8; (width * height) as usize];
        let strip = encode_jpeg_strip(&pixels, width, height, 1, 85).expect("encode");
        let tables = build_jpeg_tables(1, 85).expect("tables");
        let decoded = decode_jpeg_strip(&strip, Some(&tables)).expect("decode");
        assert_eq!(decoded.len(), pixels.len());
    }
}
