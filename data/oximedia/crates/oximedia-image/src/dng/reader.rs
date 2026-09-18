//! DNG file reader.

use crate::error::{ImageError, ImageResult};

use super::constants::*;
use super::jpeg_baseline::decode_baseline_jpeg;
use super::jpeg_lossless::{decode_lossless_jpeg, deinterleave_cfa};
use super::parser::{tiff_type_size, TiffIfd, TiffParser};
use super::types::{CfaPattern, DngCompression, DngImage, DngMetadata};

// ==========================================
// DNG Reader
// ==========================================

/// Reader for DNG (Digital Negative) files.
pub struct DngReader;

impl DngReader {
    /// Read a DNG file from raw bytes, returning the decoded image.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is not a valid DNG file or contains
    /// unsupported features.
    pub fn read(data: &[u8]) -> ImageResult<DngImage> {
        if !Self::is_dng(data) {
            return Err(ImageError::invalid_format("Not a valid DNG file"));
        }

        let (byte_order, ifds) = TiffParser::parse(data)?;
        let parser = TiffParser { byte_order };

        // Find the main raw IFD (first IFD with image data, or sub-IFD)
        let main_ifd = ifds
            .first()
            .ok_or_else(|| ImageError::invalid_format("No IFD found in DNG"))?;

        let width = parser
            .get_tag_value_u32(main_ifd, TAG_IMAGE_WIDTH, data)
            .ok_or_else(|| ImageError::invalid_format("Missing image width"))?;
        let height = parser
            .get_tag_value_u32(main_ifd, TAG_IMAGE_LENGTH, data)
            .ok_or_else(|| ImageError::invalid_format("Missing image height"))?;

        let bps = parser
            .get_tag_value_u16(main_ifd, TAG_BITS_PER_SAMPLE, data)
            .unwrap_or(16) as u8;
        let spp = parser
            .get_tag_value_u16(main_ifd, TAG_SAMPLES_PER_PIXEL, data)
            .unwrap_or(1);

        let metadata = Self::parse_metadata(&parser, main_ifd, data)?;

        // Read raw pixel data from strips
        let raw_data = Self::read_raw_strips(&parser, main_ifd, data, width, height, bps)?;

        Ok(DngImage {
            width,
            height,
            bit_depth: bps,
            channels: spp as u8,
            raw_data,
            metadata,
            is_demosaiced: false,
        })
    }

    /// Read only the metadata from a DNG file without loading pixel data.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is not a valid DNG file.
    pub fn read_metadata(data: &[u8]) -> ImageResult<DngMetadata> {
        if !Self::is_dng(data) {
            return Err(ImageError::invalid_format("Not a valid DNG file"));
        }

        let (byte_order, ifds) = TiffParser::parse(data)?;
        let parser = TiffParser { byte_order };
        let main_ifd = ifds
            .first()
            .ok_or_else(|| ImageError::invalid_format("No IFD found in DNG"))?;

        Self::parse_metadata(&parser, main_ifd, data)
    }

    /// Check if the given data appears to be a valid DNG file.
    ///
    /// Checks for TIFF magic bytes and the presence of the DNG version tag.
    #[must_use]
    pub fn is_dng(data: &[u8]) -> bool {
        if data.len() < 8 {
            return false;
        }

        // Check TIFF magic
        let is_tiff = matches!((data[0], data[1]), (0x49, 0x49) | (0x4D, 0x4D));
        if !is_tiff {
            return false;
        }

        // Check TIFF version
        let version = match (data[0], data[1]) {
            (0x49, 0x49) => u16::from_le_bytes([data[2], data[3]]),
            _ => u16::from_be_bytes([data[2], data[3]]),
        };
        if version != 42 {
            return false;
        }

        // Look for DNG version tag in the data
        // TAG_DNG_VERSION = 50706 = 0xC612
        // Search for this tag in the IFD entries
        let parse_result = TiffParser::parse(data);
        if let Ok((byte_order, ifds)) = parse_result {
            let parser = TiffParser { byte_order };
            for ifd in &ifds {
                if ifd.entries.iter().any(|e| e.tag == TAG_DNG_VERSION) {
                    // Verify the DNG version bytes look valid
                    if let Some(raw_bytes) = parser.get_tag_raw_bytes(ifd, TAG_DNG_VERSION, data) {
                        if raw_bytes.len() >= 4 && raw_bytes[0] >= 1 && raw_bytes[0] <= 2 {
                            return true;
                        }
                    }
                    // Also check inline (count=4, BYTE type fits in 4 bytes)
                    if let Some(entry) = ifd.entries.iter().find(|e| e.tag == TAG_DNG_VERSION) {
                        if entry.data_type == 1 && entry.count == 4 {
                            // BYTE type, 4 values, stored inline
                            let bytes = entry.value_offset.to_le_bytes();
                            if bytes[0] >= 1 && bytes[0] <= 2 {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        false
    }

    pub(crate) fn parse_metadata(
        parser: &TiffParser,
        ifd: &TiffIfd,
        data: &[u8],
    ) -> ImageResult<DngMetadata> {
        let mut metadata = DngMetadata::default();

        // DNG version
        if let Some(entry) = ifd.entries.iter().find(|e| e.tag == TAG_DNG_VERSION) {
            if entry.data_type == 1 && entry.count == 4 {
                let type_size = tiff_type_size(entry.data_type);
                let total_size = type_size * entry.count as usize;
                if total_size <= 4 {
                    let bytes = entry.value_offset.to_le_bytes();
                    metadata.dng_version = bytes;
                } else if let Some(raw_bytes) = parser.get_tag_raw_bytes(ifd, TAG_DNG_VERSION, data)
                {
                    if raw_bytes.len() >= 4 {
                        metadata.dng_version =
                            [raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3]];
                    }
                }
            }
        }

        // Camera model
        if let Some(model) = parser.get_tag_value_string(ifd, TAG_UNIQUE_CAMERA_MODEL, data) {
            metadata.camera_model = model;
        }

        // CFA pattern
        if let Some(raw_bytes) = parser.get_tag_raw_bytes(ifd, TAG_CFA_PATTERN, data) {
            metadata.cfa_pattern = Self::parse_cfa_pattern(raw_bytes)?;
        } else if let Some(entry) = ifd.entries.iter().find(|e| e.tag == TAG_CFA_PATTERN) {
            // CFA pattern might be inline (4 bytes fit in value_offset)
            let bytes = entry.value_offset.to_le_bytes();
            if entry.count == 4 {
                metadata.cfa_pattern = Self::parse_cfa_pattern(&bytes)?;
            }
        }

        // White balance (as-shot neutral)
        if let Some(vals) = parser.get_tag_values_f64(ifd, TAG_AS_SHOT_NEUTRAL, data) {
            if vals.len() >= 3 {
                metadata.white_balance.as_shot_neutral = [vals[0], vals[1], vals[2]];
            }
        }

        // Color matrix 1
        if let Some(vals) = parser.get_tag_values_f64(ifd, TAG_COLOR_MATRIX_1, data) {
            if vals.len() >= 9 {
                metadata.color_calibration.color_matrix_1 = [
                    [vals[0], vals[1], vals[2]],
                    [vals[3], vals[4], vals[5]],
                    [vals[6], vals[7], vals[8]],
                ];
            }
        }

        // Color matrix 2
        if let Some(vals) = parser.get_tag_values_f64(ifd, TAG_COLOR_MATRIX_2, data) {
            if vals.len() >= 9 {
                metadata.color_calibration.color_matrix_2 = Some([
                    [vals[0], vals[1], vals[2]],
                    [vals[3], vals[4], vals[5]],
                    [vals[6], vals[7], vals[8]],
                ]);
            }
        }

        // Forward matrix 1
        if let Some(vals) = parser.get_tag_values_f64(ifd, TAG_FORWARD_MATRIX_1, data) {
            if vals.len() >= 9 {
                metadata.color_calibration.forward_matrix_1 = Some([
                    [vals[0], vals[1], vals[2]],
                    [vals[3], vals[4], vals[5]],
                    [vals[6], vals[7], vals[8]],
                ]);
            }
        }

        // Calibration illuminants
        if let Some(ill) = parser.get_tag_value_u16(ifd, TAG_CALIBRATION_ILLUMINANT_1, data) {
            metadata.color_calibration.illuminant_1 = ill;
        }
        if let Some(ill) = parser.get_tag_value_u16(ifd, TAG_CALIBRATION_ILLUMINANT_2, data) {
            metadata.color_calibration.illuminant_2 = Some(ill);
        }

        // Black level
        if let Some(vals) = parser.get_tag_values_f64(ifd, TAG_BLACK_LEVEL, data) {
            if !vals.is_empty() {
                metadata.black_level = vals;
            }
        }

        // White level
        if let Some(vals) = parser.get_tag_values_u32(ifd, TAG_WHITE_LEVEL, data) {
            if !vals.is_empty() {
                metadata.white_level = vals;
            }
        }

        // Active area
        if let Some(vals) = parser.get_tag_values_u32(ifd, TAG_ACTIVE_AREA, data) {
            if vals.len() >= 4 {
                metadata.active_area = Some([vals[0], vals[1], vals[2], vals[3]]);
            }
        }

        Ok(metadata)
    }

    pub(crate) fn parse_cfa_pattern(pattern: &[u8]) -> ImageResult<CfaPattern> {
        if pattern.len() < 4 {
            return Err(ImageError::invalid_format(
                "CFA pattern must be at least 4 bytes",
            ));
        }

        match [pattern[0], pattern[1], pattern[2], pattern[3]] {
            [0, 1, 1, 2] => Ok(CfaPattern::Rggb),
            [2, 1, 1, 0] => Ok(CfaPattern::Bggr),
            [1, 0, 2, 1] => Ok(CfaPattern::Grbg),
            [1, 2, 0, 1] => Ok(CfaPattern::Gbrg),
            other => Err(ImageError::unsupported(format!(
                "Unknown CFA pattern: {:?}",
                other
            ))),
        }
    }

    fn read_raw_strips(
        parser: &TiffParser,
        ifd: &TiffIfd,
        data: &[u8],
        width: u32,
        height: u32,
        bps: u8,
    ) -> ImageResult<Vec<u16>> {
        let pixel_count = width as usize * height as usize;

        // Get compression
        let compression_val = parser
            .get_tag_value_u16(ifd, TAG_COMPRESSION, data)
            .unwrap_or(1);
        let compression = DngCompression::from_u16(compression_val)?;

        // Get strip offsets and byte counts
        let strip_offsets = parser
            .get_tag_values_u32(ifd, TAG_STRIP_OFFSETS, data)
            .ok_or_else(|| ImageError::invalid_format("Missing strip offsets in DNG"))?;
        let strip_byte_counts = parser
            .get_tag_values_u32(ifd, TAG_STRIP_BYTE_COUNTS, data)
            .ok_or_else(|| ImageError::invalid_format("Missing strip byte counts in DNG"))?;

        if strip_offsets.len() != strip_byte_counts.len() {
            return Err(ImageError::invalid_format(
                "Mismatched strip offsets and byte counts",
            ));
        }

        // JPEG-based compressions decode directly to 16-bit samples; the bit
        // unpacker is bypassed entirely. Each strip is an independent JPEG
        // datastream covering a horizontal band of `rows_per_strip` rows.
        match compression {
            DngCompression::LosslessJpeg => {
                return Self::decode_lossless_jpeg_strips(
                    parser,
                    ifd,
                    data,
                    width,
                    height,
                    &strip_offsets,
                    &strip_byte_counts,
                );
            }
            DngCompression::LossyDng => {
                return Self::decode_lossy_dng_strips(
                    data,
                    width,
                    height,
                    &strip_offsets,
                    &strip_byte_counts,
                );
            }
            DngCompression::Uncompressed | DngCompression::Deflate => {}
        }

        // Read all strips into a contiguous buffer
        let mut raw_bytes = Vec::new();
        for (offset, count) in strip_offsets.iter().zip(&strip_byte_counts) {
            let off = *offset as usize;
            let cnt = *count as usize;
            if off + cnt > data.len() {
                return Err(ImageError::invalid_format("Strip data extends beyond file"));
            }
            let strip_data = &data[off..off + cnt];

            match compression {
                DngCompression::Uncompressed => {
                    raw_bytes.extend_from_slice(strip_data);
                }
                DngCompression::Deflate => {
                    let decompressed = decompress_deflate_dng(strip_data)?;
                    raw_bytes.extend_from_slice(&decompressed);
                }
                DngCompression::LosslessJpeg | DngCompression::LossyDng => unreachable!(
                    "JPEG-based compressions are handled before the byte-concatenation path"
                ),
            }
        }

        // Unpack bits to u16
        Self::unpack_bits(&raw_bytes, bps, pixel_count)
    }

    /// Decode lossless-JPEG (TIFF compression 7) strips into a CFA raster.
    ///
    /// Each strip is a complete lossless-JPEG datastream (ITU-T T.81 Annex H,
    /// process 14). Adobe's DNG converter packs the Bayer mosaic into a
    /// 2-component-per-pixel image whose width is `cfa_width / components`;
    /// [`deinterleave_cfa`] reverses that packing. The decoded `u16` samples
    /// are placed row-by-row into the full-image raster.
    fn decode_lossless_jpeg_strips(
        parser: &TiffParser,
        ifd: &TiffIfd,
        data: &[u8],
        width: u32,
        height: u32,
        strip_offsets: &[u32],
        strip_byte_counts: &[u32],
    ) -> ImageResult<Vec<u16>> {
        let width_usize = width as usize;
        let height_usize = height as usize;
        let mut raster = vec![0u16; width_usize * height_usize];

        // RowsPerStrip determines where each strip's band begins. When the tag
        // is absent the whole image is a single strip.
        let rows_per_strip = parser
            .get_tag_value_u32(ifd, TAG_ROWS_PER_STRIP, data)
            .filter(|&r| r > 0)
            .unwrap_or(height) as usize;

        for (strip_index, (offset, count)) in strip_offsets
            .iter()
            .zip(strip_byte_counts.iter())
            .enumerate()
        {
            let off = *offset as usize;
            let cnt = *count as usize;
            if off + cnt > data.len() {
                return Err(ImageError::invalid_format(
                    "Lossless JPEG strip extends beyond file",
                ));
            }
            let strip_bytes = &data[off..off + cnt];
            let image = decode_lossless_jpeg(strip_bytes)?;

            let row_start = strip_index * rows_per_strip;
            if row_start >= height_usize {
                break;
            }
            let band_rows = rows_per_strip.min(height_usize - row_start);
            // De-interleave the strip's component packing into CFA columns.
            let cfa = deinterleave_cfa(&image, width, band_rows as u32)?;
            let band_start = row_start * width_usize;
            let band_len = band_rows * width_usize;
            raster[band_start..band_start + band_len].copy_from_slice(&cfa[..band_len]);
        }

        Ok(raster)
    }

    /// Decode lossy-DNG (TIFF compression 34892) strips into a sample raster.
    ///
    /// DNG 1.4 lossy stores gamma-encoded raw data as a baseline 8-bit DCT
    /// JPEG. The decoded 8-bit luma samples are widened to `u16`; the inverse
    /// gamma is applied later by the opcode/scale pipeline. Single-component
    /// JPEGs map directly to the CFA mosaic; multi-component JPEGs contribute
    /// their first plane (DNG lossy tiles are single-component mosaics).
    fn decode_lossy_dng_strips(
        data: &[u8],
        width: u32,
        height: u32,
        strip_offsets: &[u32],
        strip_byte_counts: &[u32],
    ) -> ImageResult<Vec<u16>> {
        let width_usize = width as usize;
        let height_usize = height as usize;
        let mut raster = vec![0u16; width_usize * height_usize];
        let mut row_cursor = 0usize;

        for (offset, count) in strip_offsets.iter().zip(strip_byte_counts.iter()) {
            let off = *offset as usize;
            let cnt = *count as usize;
            if off + cnt > data.len() {
                return Err(ImageError::invalid_format(
                    "Lossy DNG strip extends beyond file",
                ));
            }
            let strip_bytes = &data[off..off + cnt];
            let image = decode_baseline_jpeg(strip_bytes)?;
            let plane = image.planes.first().ok_or_else(|| {
                ImageError::compression("Lossy DNG: baseline JPEG produced no planes")
            })?;

            let plane_w = plane.width as usize;
            let plane_h = plane.height as usize;
            let copy_w = plane_w.min(width_usize);
            for y in 0..plane_h {
                let dst_row = row_cursor + y;
                if dst_row >= height_usize {
                    break;
                }
                let src = y * plane_w;
                let dst = dst_row * width_usize;
                for x in 0..copy_w {
                    raster[dst + x] = u16::from(plane.samples[src + x]);
                }
            }
            row_cursor += plane_h;
            if row_cursor >= height_usize {
                break;
            }
        }

        Ok(raster)
    }

    /// Unpack packed bit data into u16 values.
    ///
    /// Handles 8, 10, 12, 14, and 16-bit packed formats.
    pub(crate) fn unpack_bits(
        data: &[u8],
        bit_depth: u8,
        pixel_count: usize,
    ) -> ImageResult<Vec<u16>> {
        match bit_depth {
            8 => {
                let mut result = Vec::with_capacity(pixel_count);
                for i in 0..pixel_count.min(data.len()) {
                    result.push(u16::from(data[i]));
                }
                Ok(result)
            }
            16 => {
                let mut result = Vec::with_capacity(pixel_count);
                for i in 0..pixel_count {
                    let offset = i * 2;
                    if offset + 1 >= data.len() {
                        break;
                    }
                    // DNG typically uses little-endian for pixel data
                    result.push(u16::from_le_bytes([data[offset], data[offset + 1]]));
                }
                Ok(result)
            }
            10 => Self::unpack_10bit(data, pixel_count),
            12 => Self::unpack_12bit(data, pixel_count),
            14 => Self::unpack_14bit(data, pixel_count),
            _ => Err(ImageError::unsupported(format!(
                "Bit depth {bit_depth} not supported"
            ))),
        }
    }

    /// Unpack 10-bit packed data.
    /// Every 5 bytes contain 4 pixels of 10 bits each.
    fn unpack_10bit(data: &[u8], pixel_count: usize) -> ImageResult<Vec<u16>> {
        let mut result = Vec::with_capacity(pixel_count);
        let mut bit_buffer: u64 = 0;
        let mut bits_in_buffer: u32 = 0;
        let mut byte_idx = 0;

        for _ in 0..pixel_count {
            while bits_in_buffer < 10 {
                if byte_idx >= data.len() {
                    // Pad with zeros
                    bit_buffer <<= 8;
                    bits_in_buffer += 8;
                } else {
                    bit_buffer = (bit_buffer << 8) | u64::from(data[byte_idx]);
                    bits_in_buffer += 8;
                    byte_idx += 1;
                }
            }
            bits_in_buffer -= 10;
            let value = ((bit_buffer >> bits_in_buffer) & 0x3FF) as u16;
            result.push(value);
        }

        Ok(result)
    }

    /// Unpack 12-bit packed data.
    /// Every 3 bytes contain 2 pixels of 12 bits each.
    fn unpack_12bit(data: &[u8], pixel_count: usize) -> ImageResult<Vec<u16>> {
        let mut result = Vec::with_capacity(pixel_count);
        let mut bit_buffer: u64 = 0;
        let mut bits_in_buffer: u32 = 0;
        let mut byte_idx = 0;

        for _ in 0..pixel_count {
            while bits_in_buffer < 12 {
                if byte_idx >= data.len() {
                    bit_buffer <<= 8;
                    bits_in_buffer += 8;
                } else {
                    bit_buffer = (bit_buffer << 8) | u64::from(data[byte_idx]);
                    bits_in_buffer += 8;
                    byte_idx += 1;
                }
            }
            bits_in_buffer -= 12;
            let value = ((bit_buffer >> bits_in_buffer) & 0xFFF) as u16;
            result.push(value);
        }

        Ok(result)
    }

    /// Unpack 14-bit packed data.
    /// Every 7 bytes contain 4 pixels of 14 bits each.
    fn unpack_14bit(data: &[u8], pixel_count: usize) -> ImageResult<Vec<u16>> {
        let mut result = Vec::with_capacity(pixel_count);
        let mut bit_buffer: u64 = 0;
        let mut bits_in_buffer: u32 = 0;
        let mut byte_idx = 0;

        for _ in 0..pixel_count {
            while bits_in_buffer < 14 {
                if byte_idx >= data.len() {
                    bit_buffer <<= 8;
                    bits_in_buffer += 8;
                } else {
                    bit_buffer = (bit_buffer << 8) | u64::from(data[byte_idx]);
                    bits_in_buffer += 8;
                    byte_idx += 1;
                }
            }
            bits_in_buffer -= 14;
            let value = ((bit_buffer >> bits_in_buffer) & 0x3FFF) as u16;
            result.push(value);
        }

        Ok(result)
    }
}

fn decompress_deflate_dng(data: &[u8]) -> ImageResult<Vec<u8>> {
    use oxiarc_deflate::ZlibStreamDecoder;
    use std::io::Read;

    let mut decoder = ZlibStreamDecoder::new(data);
    let mut output = Vec::new();
    decoder
        .read_to_end(&mut output)
        .map_err(|e| ImageError::Compression(format!("DNG deflate decompression failed: {e}")))?;
    Ok(output)
}
