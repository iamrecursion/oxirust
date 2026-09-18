//! Binary serialisation helpers for OpenEXR 2.0 multi-part files.
//!
//! Implements the write path: string I/O, attribute encoding, scanline
//! pixel encoding, compression dispatch, and the top-level multi-part
//! document writer.

use byteorder::{LittleEndian, WriteBytesExt};
use half::f16;
use std::io::{Cursor, Seek, SeekFrom, Write};

use super::types::{ExrBox2i, ExrChannel, ExrChannelType, ExrCompression, ExrPart};
use super::{MultiPartExr, EXR_MAGIC, EXR_VERSION, VERSION_FLAG_MULTIPART};
use crate::error::ImageResult;

// ── String I/O ────────────────────────────────────────────────────────────────

/// Write a NUL-terminated UTF-8 string.
pub(super) fn write_nul_string<W: Write>(w: &mut W, s: &str) -> ImageResult<()> {
    w.write_all(s.as_bytes())?;
    w.write_u8(0)?;
    Ok(())
}

// ── Attribute I/O ─────────────────────────────────────────────────────────────

/// Write a generic attribute: `name\0 type\0 size(u32) data`.
fn write_attr<W, F>(w: &mut W, name: &str, attr_type: &str, data_fn: F) -> ImageResult<()>
where
    W: Write,
    F: FnOnce(&mut Vec<u8>) -> ImageResult<()>,
{
    write_nul_string(w, name)?;
    write_nul_string(w, attr_type)?;
    let mut buf = Vec::new();
    data_fn(&mut buf)?;
    w.write_u32::<LittleEndian>(buf.len() as u32)?;
    w.write_all(&buf)?;
    Ok(())
}

/// Write the `channels` (chlist) attribute for a single part.
fn write_channels_attr<W: Write>(w: &mut W, channels: &[ExrChannel]) -> ImageResult<()> {
    write_attr(w, "channels", "chlist", |buf| {
        for ch in channels {
            write_nul_string(buf, &ch.name)?;
            buf.write_u32::<LittleEndian>(ch.channel_type.wire_code())?;
            buf.write_u8(if ch.linear { 1 } else { 0 })?; // pLinear
            buf.write_all(&[0u8; 3])?; // reserved
            buf.write_u32::<LittleEndian>(ch.x_sampling)?;
            buf.write_u32::<LittleEndian>(ch.y_sampling)?;
        }
        write_nul_string(buf, "")?; // end of chlist
        Ok(())
    })
}

/// Write a box2i attribute.
fn write_box2i_attr<W: Write>(w: &mut W, name: &str, r: &ExrBox2i) -> ImageResult<()> {
    write_attr(w, name, "box2i", |buf| {
        buf.write_i32::<LittleEndian>(r.x_min)?;
        buf.write_i32::<LittleEndian>(r.y_min)?;
        buf.write_i32::<LittleEndian>(r.x_max)?;
        buf.write_i32::<LittleEndian>(r.y_max)?;
        Ok(())
    })
}

/// Write a string attribute.
fn write_string_attr<W: Write>(w: &mut W, name: &str, value: &str) -> ImageResult<()> {
    write_attr(w, name, "string", |buf| {
        buf.write_all(value.as_bytes())?;
        Ok(())
    })
}

/// Write a u8 attribute with a custom type name (e.g. `"compression"`).
fn write_u8_attr<W: Write>(w: &mut W, name: &str, type_name: &str, value: u8) -> ImageResult<()> {
    write_attr(w, name, type_name, |buf| {
        buf.write_u8(value)?;
        Ok(())
    })
}

/// Write a f32 attribute.
fn write_f32_attr<W: Write>(w: &mut W, name: &str, type_name: &str, value: f32) -> ImageResult<()> {
    write_attr(w, name, type_name, |buf| {
        buf.write_f32::<LittleEndian>(value)?;
        Ok(())
    })
}

/// Write a v2f attribute.
fn write_v2f_attr<W: Write>(w: &mut W, name: &str, x: f32, y: f32) -> ImageResult<()> {
    write_attr(w, name, "v2f", |buf| {
        buf.write_f32::<LittleEndian>(x)?;
        buf.write_f32::<LittleEndian>(y)?;
        Ok(())
    })
}

// ── Per-part header writer ─────────────────────────────────────────────────────

/// Serialise a single part header (without the final NUL terminator for the
/// inter-part gap — that is written by the caller).
fn write_part_header<W: Write>(w: &mut W, part: &ExrPart) -> ImageResult<()> {
    // Mandatory attributes in alphabetical order (OpenEXR spec recommends this)
    write_channels_attr(w, &part.channels)?;
    write_u8_attr(
        w,
        "compression",
        "compression",
        part.compression.wire_code(),
    )?;
    write_box2i_attr(w, "dataWindow", &part.data_window)?;
    write_box2i_attr(w, "displayWindow", &part.display_window)?;
    write_u8_attr(w, "lineOrder", "lineOrder", 0)?; // IncreasingY
    write_string_attr(w, "name", &part.name)?;
    write_f32_attr(w, "pixelAspectRatio", "float", 1.0)?;
    write_v2f_attr(w, "screenWindowCenter", 0.0, 0.0)?;
    write_f32_attr(w, "screenWindowWidth", "float", 1.0)?;
    write_string_attr(w, "type", part.part_type.as_str())?;
    // End of this part's header
    w.write_u8(0)?;
    Ok(())
}

// ── Pixel encoding ────────────────────────────────────────────────────────────

/// Convert a scanline's interleaved f32 pixel data to the EXR wire format.
///
/// EXR stores scanlines in **channel-major** (non-interleaved) order: all
/// samples for channel 0, then all samples for channel 1, etc.  Within each
/// channel block the samples are in x-order.
///
/// The output byte layout for one scanline is:
/// ```text
/// [ch0_px0, ch0_px1, …, ch0_pxW-1, ch1_px0, …, chN_pxW-1]
/// ```
/// where each sample is encoded in the channel's declared type.
pub(super) fn encode_scanline(pixels: &[f32], width: u32, channels: &[ExrChannel]) -> Vec<u8> {
    let w = width as usize;
    let n_ch = channels.len();
    let mut out = Vec::with_capacity(
        channels
            .iter()
            .map(|c| c.channel_type.bytes_per_sample() * w)
            .sum::<usize>(),
    );

    for (ch_idx, ch) in channels.iter().enumerate() {
        match ch.channel_type {
            ExrChannelType::Half => {
                for px in 0..w {
                    let sample = pixels[px * n_ch + ch_idx];
                    let half_bits = f16::from_f32(sample).to_bits();
                    out.extend_from_slice(&half_bits.to_le_bytes());
                }
            }
            ExrChannelType::Float => {
                for px in 0..w {
                    let sample = pixels[px * n_ch + ch_idx];
                    out.extend_from_slice(&sample.to_le_bytes());
                }
            }
            ExrChannelType::Uint => {
                for px in 0..w {
                    let sample = pixels[px * n_ch + ch_idx];
                    // Clamp and convert f32 → u32 (normalised [0,1] range)
                    let clamped = sample.clamp(0.0, u32::MAX as f32);
                    let uint_val = clamped as u32;
                    out.extend_from_slice(&uint_val.to_le_bytes());
                }
            }
        }
    }
    out
}

// ── Compression dispatch ──────────────────────────────────────────────────────

pub(super) fn apply_compression(raw: &[u8], compression: ExrCompression) -> ImageResult<Vec<u8>> {
    use crate::exr::{
        compress_b44, compress_b44a, compress_dwaa, compress_dwab, compress_piz, compress_pxr24,
        compress_rle, compress_zip,
    };
    match compression {
        ExrCompression::None => Ok(raw.to_vec()),
        ExrCompression::Rle => compress_rle(raw),
        ExrCompression::ZipSingle | ExrCompression::Zip => compress_zip(raw),
        ExrCompression::Piz => compress_piz(raw),
        ExrCompression::Pxr24 => compress_pxr24(raw),
        ExrCompression::B44 => compress_b44(raw),
        ExrCompression::B44a => compress_b44a(raw),
        ExrCompression::Dwaa => compress_dwaa(raw),
        ExrCompression::Dwab => compress_dwab(raw),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Top-level writer
// ═════════════════════════════════════════════════════════════════════════════

/// Encode the entire multi-part EXR document to bytes.
pub(super) fn write_multipart_exr(doc: &MultiPartExr) -> ImageResult<Vec<u8>> {
    // Use a Cursor<Vec<u8>> so we can seek back to fill offset tables.
    let mut cur: Cursor<Vec<u8>> = Cursor::new(Vec::new());

    // ── File header ───────────────────────────────────────────────────────────
    cur.write_u32::<LittleEndian>(EXR_MAGIC)?;
    // Version word: version byte | (flags << 8).  Multi-part flag = 0x1000.
    let version_word: u32 = (EXR_VERSION as u32) | ((VERSION_FLAG_MULTIPART) << 8);
    cur.write_u32::<LittleEndian>(version_word)?;

    // ── Part headers ─────────────────────────────────────────────────────────
    for part in &doc.parts {
        write_part_header(&mut cur, part)?;
    }
    // End of all headers (second consecutive NUL, per spec)
    cur.write_u8(0)?;

    // ── Chunk offset tables (one table per part) ───────────────────────────
    //
    // The offset table for part i has one u64 per *chunk*.
    // For ScanlineImage with `scanlines_per_block = 1`, the number of chunks
    // equals the height.
    //
    // We write placeholder zeroes and come back to fill them in after writing
    // the chunks.
    let mut offset_table_positions: Vec<u64> = Vec::with_capacity(doc.parts.len());
    let mut chunk_counts: Vec<u32> = Vec::with_capacity(doc.parts.len());

    for part in &doc.parts {
        let n_chunks = part.height; // scanlines_per_block = 1 for our supported types
        chunk_counts.push(n_chunks);

        let table_pos = cur.stream_position()?;
        offset_table_positions.push(table_pos);

        for _ in 0..n_chunks {
            cur.write_u64::<LittleEndian>(0)?;
        }
    }

    // ── Chunk data ────────────────────────────────────────────────────────────
    //
    // For multi-part files, each chunk is prefixed with `part_number: i32`.
    // Then: `y_coordinate: i32`, `pixel_data_size: u32`, `pixel_data: [u8]`.
    let mut chunk_offsets: Vec<Vec<u64>> = Vec::with_capacity(doc.parts.len());

    for (part_idx, part) in doc.parts.iter().enumerate() {
        let n_ch = part.channels.len();
        let width = part.width;
        let height = part.height;
        let n_chunks = chunk_counts[part_idx] as usize;
        let mut offsets: Vec<u64> = Vec::with_capacity(n_chunks);

        for y in 0..height as usize {
            let chunk_pos = cur.stream_position()?;
            offsets.push(chunk_pos);

            // part_number (i32 LE) — multi-part header field
            cur.write_i32::<LittleEndian>(part_idx as i32)?;

            // y coordinate
            let abs_y = (part.data_window.y_min + y as i32) as i32;
            cur.write_i32::<LittleEndian>(abs_y)?;

            // Gather one scanline's worth of interleaved f32
            let row_start = y * width as usize * n_ch;
            let row_end = row_start + width as usize * n_ch;
            let row_pixels = &part.pixels[row_start..row_end];

            // Encode to wire bytes (channel-major, typed)
            let raw = encode_scanline(row_pixels, width, &part.channels);

            // Apply compression
            let compressed = apply_compression(&raw, part.compression)?;

            // pixel_data_size
            cur.write_u32::<LittleEndian>(compressed.len() as u32)?;
            cur.write_all(&compressed)?;
        }

        chunk_offsets.push(offsets);
    }

    // ── Back-fill offset tables ───────────────────────────────────────────────
    for (part_idx, table_pos) in offset_table_positions.iter().enumerate() {
        cur.seek(SeekFrom::Start(*table_pos))?;
        for &offset in &chunk_offsets[part_idx] {
            cur.write_u64::<LittleEndian>(offset)?;
        }
    }

    Ok(cur.into_inner())
}
