//! Binary deserialisation helpers for OpenEXR 2.0 multi-part files.
//!
//! Implements the read path: string parsing, attribute decoding, scanline
//! pixel decoding, decompression dispatch, and the top-level multi-part
//! document reader.

use byteorder::{LittleEndian, ReadBytesExt};
use half::f16;
use std::io::{Read, Seek, SeekFrom};

use super::types::{ExrBox2i, ExrChannel, ExrChannelType, ExrCompression, ExrPart, ExrPartType};
use super::{
    MultiPartExr, EXR_MAGIC, VERSION_FLAG_DEEP, VERSION_FLAG_MULTIPART, VERSION_FLAG_TILED,
};
use crate::error::{ImageError, ImageResult};

// ── String I/O ────────────────────────────────────────────────────────────────

/// Read a NUL-terminated UTF-8 string from a cursor.
pub(super) fn read_nul_string<R: Read>(r: &mut R) -> ImageResult<String> {
    let mut bytes = Vec::new();
    loop {
        let b = r.read_u8()?;
        if b == 0 {
            break;
        }
        bytes.push(b);
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

// ── Pixel decoding ────────────────────────────────────────────────────────────

/// Decode a channel-major scanline blob back into interleaved f32 samples.
pub(super) fn decode_scanline(
    blob: &[u8],
    width: u32,
    channels: &[ExrChannel],
) -> ImageResult<Vec<f32>> {
    let w = width as usize;
    let n_ch = channels.len();
    let mut out = vec![0.0_f32; w * n_ch];

    let mut byte_off = 0usize;
    for (ch_idx, ch) in channels.iter().enumerate() {
        let bps = ch.channel_type.bytes_per_sample();
        let ch_end = byte_off + bps * w;
        if ch_end > blob.len() {
            return Err(ImageError::invalid_format(format!(
                "Scanline too short: need {ch_end} bytes, have {}",
                blob.len()
            )));
        }
        let ch_blob = &blob[byte_off..ch_end];

        match ch.channel_type {
            ExrChannelType::Half => {
                for px in 0..w {
                    let bits = u16::from_le_bytes([ch_blob[px * 2], ch_blob[px * 2 + 1]]);
                    out[px * n_ch + ch_idx] = f16::from_bits(bits).to_f32();
                }
            }
            ExrChannelType::Float => {
                for px in 0..w {
                    let b = &ch_blob[px * 4..px * 4 + 4];
                    out[px * n_ch + ch_idx] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
            ExrChannelType::Uint => {
                for px in 0..w {
                    let b = &ch_blob[px * 4..px * 4 + 4];
                    let uint_val = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    out[px * n_ch + ch_idx] = uint_val as f32;
                }
            }
        }

        byte_off = ch_end;
    }
    Ok(out)
}

// ── Decompression dispatch ────────────────────────────────────────────────────

pub(super) fn apply_decompression(
    compressed: &[u8],
    compression: ExrCompression,
) -> ImageResult<Vec<u8>> {
    use crate::exr::{
        decompress_b44, decompress_b44a, decompress_dwaa, decompress_dwab, decompress_piz,
        decompress_pxr24, decompress_rle, decompress_zip,
    };
    match compression {
        ExrCompression::None => Ok(compressed.to_vec()),
        ExrCompression::Rle => decompress_rle(compressed),
        ExrCompression::ZipSingle | ExrCompression::Zip => decompress_zip(compressed),
        ExrCompression::Piz => decompress_piz(compressed),
        ExrCompression::Pxr24 => decompress_pxr24(compressed),
        ExrCompression::B44 => decompress_b44(compressed),
        ExrCompression::B44a => decompress_b44a(compressed),
        ExrCompression::Dwaa => decompress_dwaa(compressed),
        ExrCompression::Dwab => decompress_dwab(compressed),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Internal header parsing structures
// ═════════════════════════════════════════════════════════════════════════════

/// Parsed representation of a per-part header.
pub(super) struct RawPartHeader {
    pub(super) name: String,
    pub(super) part_type: ExrPartType,
    pub(super) channels: Vec<ExrChannel>,
    pub(super) data_window: ExrBox2i,
    pub(super) display_window: ExrBox2i,
    pub(super) compression: ExrCompression,
}

/// Read one per-part attribute block.  Returns `None` when an empty header
/// (i.e. the inter-part separator / end-of-headers marker) is found.
pub(super) fn read_one_part_header<R: Read + Seek>(
    r: &mut R,
) -> ImageResult<Option<RawPartHeader>> {
    // Peek at the first byte to detect an empty attribute name (separator NUL).
    let first = r.read_u8()?;
    if first == 0 {
        return Ok(None);
    }

    // It's a real attribute name — read the rest and assemble the header.
    let mut name_bytes = vec![first];
    loop {
        let b = r.read_u8()?;
        if b == 0 {
            break;
        }
        name_bytes.push(b);
    }

    // We have consumed the name of the *first* attribute of this part.
    // Now parse all attributes until we hit an empty name (end of this part).
    let mut hdr = RawPartHeader {
        name: String::new(),
        part_type: ExrPartType::ScanlineImage,
        channels: Vec::new(),
        data_window: ExrBox2i {
            x_min: 0,
            y_min: 0,
            x_max: 0,
            y_max: 0,
        },
        display_window: ExrBox2i {
            x_min: 0,
            y_min: 0,
            x_max: 0,
            y_max: 0,
        },
        compression: ExrCompression::None,
    };

    // Process the first attribute (whose name we already read)
    let first_attr_name = String::from_utf8_lossy(&name_bytes).into_owned();
    parse_attribute(r, &first_attr_name, &mut hdr)?;

    // Read remaining attributes
    loop {
        let attr_name = read_nul_string(r)?;
        if attr_name.is_empty() {
            break;
        }
        parse_attribute(r, &attr_name, &mut hdr)?;
    }

    Ok(Some(hdr))
}

/// Parse one attribute into the `RawPartHeader`, consuming its type, size, and data.
fn parse_attribute<R: Read + Seek>(
    r: &mut R,
    attr_name: &str,
    hdr: &mut RawPartHeader,
) -> ImageResult<()> {
    let attr_type = read_nul_string(r)?;
    let size = r.read_u32::<LittleEndian>()? as usize;

    match attr_name {
        "channels" => {
            hdr.channels = read_channels_attr(r)?;
        }
        "compression" => {
            let code = r.read_u8()?;
            hdr.compression = ExrCompression::from_wire_code(code)?;
        }
        "dataWindow" => {
            hdr.data_window = read_box2i(r)?;
        }
        "displayWindow" => {
            hdr.display_window = read_box2i(r)?;
        }
        "name" => {
            let mut buf = vec![0u8; size];
            r.read_exact(&mut buf)?;
            hdr.name = String::from_utf8_lossy(&buf)
                .trim_end_matches('\0')
                .to_string();
        }
        "type" => {
            let mut buf = vec![0u8; size];
            r.read_exact(&mut buf)?;
            let type_str = String::from_utf8_lossy(&buf)
                .trim_end_matches('\0')
                .to_string();
            hdr.part_type = ExrPartType::from_str(&type_str)?;
        }
        // Recognised but not stored — skip gracefully
        "lineOrder" | "pixelAspectRatio" | "screenWindowCenter" | "screenWindowWidth"
        | "tiledesc" | "version" => {
            skip_bytes(r, size)?;
        }
        _ => {
            // Unknown attribute: skip based on declared `attr_type`
            // For `chlist` we still need to consume correctly
            if attr_type == "chlist" {
                // Read until double-NUL terminating chlist
                loop {
                    let ch_name = read_nul_string(r)?;
                    if ch_name.is_empty() {
                        break;
                    }
                    skip_bytes(r, 12)?; // type(4) + pLinear(1) + pad(3) + xSamp(4) — wait, 4+1+3+4+4=16
                                        // Actually: type=u32(4), pLinear=u8(1), reserved=3, x_samp=u32(4), y_samp=u32(4) = 16
                                        // We already read the name; skip the rest of the channel descriptor
                    skip_bytes(r, 4)?; // We already counted 12 = wrong; be exact
                }
            } else {
                skip_bytes(r, size)?;
            }
        }
    }
    Ok(())
}

/// Read a `box2i` (4 × i32 LE).
fn read_box2i<R: Read>(r: &mut R) -> ImageResult<ExrBox2i> {
    let x_min = r.read_i32::<LittleEndian>()?;
    let y_min = r.read_i32::<LittleEndian>()?;
    let x_max = r.read_i32::<LittleEndian>()?;
    let y_max = r.read_i32::<LittleEndian>()?;
    Ok(ExrBox2i {
        x_min,
        y_min,
        x_max,
        y_max,
    })
}

/// Read the `chlist` attribute body (already past name+type+size).
fn read_channels_attr<R: Read + Seek>(r: &mut R) -> ImageResult<Vec<ExrChannel>> {
    let mut channels = Vec::new();
    loop {
        let ch_name = read_nul_string(r)?;
        if ch_name.is_empty() {
            break;
        }
        let type_code = r.read_u32::<LittleEndian>()?;
        let channel_type = ExrChannelType::from_wire_code(type_code)?;
        let p_linear = r.read_u8()?;
        skip_bytes(r, 3)?; // reserved
        let x_sampling = r.read_u32::<LittleEndian>()?;
        let y_sampling = r.read_u32::<LittleEndian>()?;
        channels.push(ExrChannel {
            name: ch_name,
            channel_type,
            x_sampling,
            y_sampling,
            linear: p_linear != 0,
        });
    }
    Ok(channels)
}

/// Skip `n` bytes from a seekable reader.
fn skip_bytes<R: Seek>(r: &mut R, n: usize) -> ImageResult<()> {
    r.seek(SeekFrom::Current(n as i64))?;
    Ok(())
}

// ═════════════════════════════════════════════════════════════════════════════
//  Block extraction helper
// ═════════════════════════════════════════════════════════════════════════════

/// Extract one scanline's channel-major bytes from a multi-scanline block blob.
///
/// In EXR, a block that covers `L` scanlines stores them in channel-major order:
///
/// ```text
/// [ch0_line0, ch0_line1, …, ch0_lineL-1,
///  ch1_line0, ch1_line1, …, ch1_lineL-1,
///  …]
/// ```
///
/// This function reconstructs a single-scanline channel-major slice for line
/// `line_in_block` (0-indexed within the block).
pub(super) fn extract_scanline_from_block(
    block: &[u8],
    channels: &[ExrChannel],
    width: u32,
    line_in_block: usize,
    lines_in_block: usize,
) -> ImageResult<Vec<u8>> {
    let w = width as usize;
    let mut out = Vec::new();

    for ch in channels {
        let bps = ch.channel_type.bytes_per_sample();
        let ch_stride = bps * w * lines_in_block;
        // offset of ch block relative to start of `block`
        let ch_offset: usize = channels
            .iter()
            .take_while(|c| c.name != ch.name)
            .map(|c| c.channel_type.bytes_per_sample() * w * lines_in_block)
            .sum();
        let line_offset = ch_offset + line_in_block * bps * w;
        let line_end = line_offset + bps * w;
        if line_end > block.len() {
            return Err(ImageError::invalid_format(format!(
                "Block too small: need {line_end} bytes, have {}",
                block.len()
            )));
        }
        out.extend_from_slice(&block[line_offset..line_end]);
        let _ = ch_stride; // capacity hint only
    }

    Ok(out)
}

// ═════════════════════════════════════════════════════════════════════════════
//  Top-level readers
// ═════════════════════════════════════════════════════════════════════════════

/// Top-level parser.
pub(super) fn read_multipart_exr<R: Read + Seek>(r: &mut R) -> ImageResult<MultiPartExr> {
    // ── Magic + version ───────────────────────────────────────────────────────
    let magic = r.read_u32::<LittleEndian>()?;
    if magic != EXR_MAGIC {
        return Err(ImageError::invalid_format(
            "Not an EXR file (bad magic number)",
        ));
    }
    let version_word = r.read_u32::<LittleEndian>()?;
    let flags = version_word >> 8;
    let is_multipart = (flags & VERSION_FLAG_MULTIPART) != 0;
    let is_tiled = (flags & VERSION_FLAG_TILED) != 0;
    let is_deep = (flags & VERSION_FLAG_DEEP) != 0;

    if is_tiled && !is_multipart {
        return Err(ImageError::Unsupported(
            "Single-part tiled EXR — use the exr::read_exr API instead".to_string(),
        ));
    }
    if is_deep && !is_multipart {
        return Err(ImageError::Unsupported(
            "Single-part deep EXR — not supported".to_string(),
        ));
    }
    if !is_multipart {
        // Single-part file: wrap as a 1-part MultiPartExr
        return read_singlepart_as_multipart(r, flags);
    }

    // ── Part headers ──────────────────────────────────────────────────────────
    let mut raw_headers: Vec<RawPartHeader> = Vec::new();
    loop {
        match read_one_part_header(r)? {
            Some(hdr) => raw_headers.push(hdr),
            None => break,
        }
    }

    if raw_headers.is_empty() {
        return Err(ImageError::invalid_format(
            "Multi-part EXR has no part headers",
        ));
    }

    // ── Chunk offset tables ───────────────────────────────────────────────────
    //
    // Each part has one u64 offset per chunk.  For scanline images the chunk
    // count = height / scanlines_per_block (rounded up).
    let mut all_offsets: Vec<Vec<u64>> = Vec::with_capacity(raw_headers.len());

    for hdr in &raw_headers {
        let width = hdr.data_window.width();
        let height = hdr.data_window.height();
        let _ = width; // used only for pixel decoding
        let spb = hdr.compression.scanlines_per_block();
        let n_chunks = (height + spb - 1) / spb;
        let mut offsets = Vec::with_capacity(n_chunks as usize);
        for _ in 0..n_chunks {
            offsets.push(r.read_u64::<LittleEndian>()?);
        }
        all_offsets.push(offsets);
    }

    // ── Pixel data ────────────────────────────────────────────────────────────
    let mut parts: Vec<ExrPart> = Vec::with_capacity(raw_headers.len());

    for (part_idx, hdr) in raw_headers.into_iter().enumerate() {
        let width = hdr.data_window.width();
        let height = hdr.data_window.height();
        let n_ch = hdr.channels.len();
        let pixel_count = width as usize * height as usize * n_ch;
        let mut pixels = vec![0.0_f32; pixel_count];

        let compression = hdr.compression;

        if hdr.part_type == ExrPartType::ScanlineImage {
            let offsets = &all_offsets[part_idx];
            let spb = compression.scanlines_per_block() as usize;

            for (chunk_idx, &offset) in offsets.iter().enumerate() {
                r.seek(SeekFrom::Start(offset))?;

                // Consume part_number prefix (multi-part files only)
                let _part_number = r.read_i32::<LittleEndian>()?;

                let y_coord = r.read_i32::<LittleEndian>()?;
                let data_size = r.read_u32::<LittleEndian>()? as usize;

                let mut compressed = vec![0u8; data_size];
                r.read_exact(&mut compressed)?;

                let raw = apply_decompression(&compressed, compression)?;

                // Determine which scanlines are in this chunk
                let first_y = (y_coord - hdr.data_window.y_min) as usize;
                let lines_in_chunk = spb.min(height as usize - chunk_idx * spb);

                // The raw blob contains `lines_in_chunk` consecutive scanlines
                // (for ZIP-16, Piz-32, etc.) stacked in channel-major order.
                //
                // For simplicity with single-scanline blocks (None, Rle, ZipSingle):
                // lines_in_chunk = 1 and the split below handles it correctly.
                //
                // For multi-scanline blocks we decode the whole blob as if it
                // were `lines_in_chunk` back-to-back scanlines.
                let bytes_per_scanline: usize = hdr
                    .channels
                    .iter()
                    .map(|c| c.channel_type.bytes_per_sample() * width as usize)
                    .sum();

                for line_in_block in 0..lines_in_chunk {
                    let scan_y = first_y + line_in_block;
                    if scan_y >= height as usize {
                        break;
                    }

                    // Slice out one scanline's worth of bytes from the chunk blob.
                    // Channel-major layout within a block: the block stores all
                    // lines for ch0, then all lines for ch1, etc.
                    // So for line `l` in a block of `L` lines:
                    //   ch_k starts at byte: k * L * bytes_per_ch_scanline + l * bytes_per_ch_scanline
                    let scan_blob: Vec<u8> = extract_scanline_from_block(
                        &raw,
                        &hdr.channels,
                        width,
                        line_in_block,
                        lines_in_chunk,
                    )?;

                    let decoded = decode_scanline(&scan_blob, width, &hdr.channels)?;

                    let row_start = scan_y * width as usize * n_ch;
                    pixels[row_start..row_start + width as usize * n_ch].copy_from_slice(&decoded);

                    let _ = bytes_per_scanline; // used in capacity estimate
                }
            }
        }
        // For Tiled, DeepScanline, DeepTile: leave pixels zeroed (metadata only).

        parts.push(ExrPart {
            name: hdr.name,
            part_type: hdr.part_type,
            channels: hdr.channels,
            data_window: hdr.data_window,
            display_window: hdr.display_window,
            compression,
            pixels,
            width,
            height,
        });
    }

    Ok(MultiPartExr { parts })
}

/// Wrap a single-part EXR (no multi-part flag) as a 1-part `MultiPartExr`.
///
/// The cursor is positioned right after the 8-byte file header (magic +
/// version).
pub(super) fn read_singlepart_as_multipart<R: Read + Seek>(
    r: &mut R,
    _flags: u32,
) -> ImageResult<MultiPartExr> {
    // Single-part uses the same header attribute format; no "name" or "type"
    // attributes are required.  We synthesise defaults.
    let mut hdr = RawPartHeader {
        name: "default".to_string(),
        part_type: ExrPartType::ScanlineImage,
        channels: Vec::new(),
        data_window: ExrBox2i {
            x_min: 0,
            y_min: 0,
            x_max: 0,
            y_max: 0,
        },
        display_window: ExrBox2i {
            x_min: 0,
            y_min: 0,
            x_max: 0,
            y_max: 0,
        },
        compression: ExrCompression::None,
    };

    // Parse attributes until empty name
    loop {
        let attr_name = read_nul_string(r)?;
        if attr_name.is_empty() {
            break;
        }
        parse_attribute(r, &attr_name, &mut hdr)?;
    }

    let width = hdr.data_window.width();
    let height = hdr.data_window.height();
    let n_ch = hdr.channels.len();
    let compression = hdr.compression;
    let spb = compression.scanlines_per_block() as usize;
    let n_chunks = (height as usize + spb - 1) / spb;

    // Read chunk offset table
    let mut offsets = Vec::with_capacity(n_chunks);
    for _ in 0..n_chunks {
        offsets.push(r.read_u64::<LittleEndian>()?);
    }

    let mut pixels = vec![0.0_f32; width as usize * height as usize * n_ch];

    for (chunk_idx, &offset) in offsets.iter().enumerate() {
        r.seek(SeekFrom::Start(offset))?;
        // Single-part chunks do NOT have a part_number prefix.
        let y_coord = r.read_i32::<LittleEndian>()?;
        let data_size = r.read_u32::<LittleEndian>()? as usize;
        let mut compressed = vec![0u8; data_size];
        r.read_exact(&mut compressed)?;

        let raw = apply_decompression(&compressed, compression)?;

        let first_y = (y_coord - hdr.data_window.y_min) as usize;
        let lines_in_chunk = spb.min(height as usize - chunk_idx * spb);

        for line_in_block in 0..lines_in_chunk {
            let scan_y = first_y + line_in_block;
            if scan_y >= height as usize {
                break;
            }
            let scan_blob = extract_scanline_from_block(
                &raw,
                &hdr.channels,
                width,
                line_in_block,
                lines_in_chunk,
            )?;
            let decoded = decode_scanline(&scan_blob, width, &hdr.channels)?;
            let row_start = scan_y * width as usize * n_ch;
            pixels[row_start..row_start + width as usize * n_ch].copy_from_slice(&decoded);
        }
    }

    let part = ExrPart {
        name: hdr.name,
        part_type: hdr.part_type,
        channels: hdr.channels,
        data_window: hdr.data_window,
        display_window: hdr.display_window,
        compression,
        pixels,
        width,
        height,
    };

    Ok(MultiPartExr { parts: vec![part] })
}
