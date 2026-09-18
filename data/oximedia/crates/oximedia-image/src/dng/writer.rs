//! DNG file writer.

use crate::error::{ImageError, ImageResult};

use super::constants::*;
use super::parser::tiff_type_size;
use super::types::{DngCompression, DngImage, DngMetadata};

// ==========================================
// DNG Writer
// ==========================================

/// Writer for DNG (Digital Negative) files.
pub struct DngWriter;

// ==========================================
// IFD entry model
// ==========================================
//
// Every TIFF/DNG IFD entry is a fixed 12-byte record: `tag(2) + type(2) +
// count(4) + value_or_offset(4)`. The last 4 bytes hold the value directly
// when it fits (`type_size * count <= 4`); otherwise they hold the *byte
// offset*, measured from the start of the file, of the actual value data,
// which is written out-of-line after the IFD ("deferred" data below).
//
// The bug this module fixes was a mismatch between the order used to
// compute those offsets (tag-ascending, i.e. post-sort — required by the
// TIFF spec) and the order used to physically append the deferred bytes
// (tag push order, which does not match tag-ascending order here). The fix
// below closes that gap structurally: `TagValue::Deferred` payloads are
// resolved and appended to `deferred_data` in exactly one pass, over the
// already-sorted tag list, so an entry's recorded offset is always the
// position its bytes are written to.

/// A tag's value: either small enough to live inline in the entry's 4-byte
/// value/offset field, or large enough that it must be written to the
/// out-of-line deferred data area following the IFD.
enum TagValue {
    /// Fits directly in the entry's 4-byte value/offset field.
    Inline(u32),
    /// Too large to fit inline (`type_size * count > 4`); these are the raw
    /// bytes to place in the deferred data area. The entry's on-disk value
    /// field is resolved to that data's file offset once the final,
    /// tag-sorted entry order is known.
    Deferred(Vec<u8>),
}

/// A pending TIFF/DNG IFD entry awaiting final layout.
struct PendingTag {
    tag: u16,
    dtype: u16,
    count: u32,
    value: TagValue,
}

impl PendingTag {
    /// Builds an entry whose value fits inline (`type_size * count <= 4`
    /// bytes), stored directly in the IFD entry's value/offset field.
    fn inline(tag: u16, dtype: u16, count: u32, value: u32) -> Self {
        debug_assert!(
            tiff_type_size(dtype) * count as usize <= 4,
            "DNG writer bug: tag {tag} value does not actually fit inline"
        );
        Self {
            tag,
            dtype,
            count,
            value: TagValue::Inline(value),
        }
    }

    /// Builds an entry whose value does not fit inline and must be written
    /// to the deferred (out-of-line) data area. `bytes.len()` must equal
    /// `type_size(dtype) * count`.
    fn deferred(tag: u16, dtype: u16, count: u32, bytes: Vec<u8>) -> Self {
        debug_assert!(
            tiff_type_size(dtype) * count as usize > 4,
            "DNG writer bug: tag {tag} value fits inline; use PendingTag::inline instead"
        );
        debug_assert_eq!(
            bytes.len(),
            tiff_type_size(dtype) * count as usize,
            "DNG writer bug: tag {tag} payload length does not match type_size*count"
        );
        Self {
            tag,
            dtype,
            count,
            value: TagValue::Deferred(bytes),
        }
    }
}

impl DngWriter {
    /// Write a DNG image to bytes.
    ///
    /// Writes an uncompressed DNG file with the specified metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the image data is invalid.
    pub fn write(image: &DngImage) -> ImageResult<Vec<u8>> {
        let mut output = Vec::new();

        // TIFF header (little-endian)
        output.extend_from_slice(&[0x49, 0x49]); // "II" = little-endian
        output.extend_from_slice(&42u16.to_le_bytes()); // TIFF version

        // IFD offset placeholder (will be at end of pixel data)
        let ifd_offset_pos = output.len();
        output.extend_from_slice(&0u32.to_le_bytes());

        // Write pixel data immediately after header
        let data_offset = output.len() as u32;
        let pixel_bytes = Self::pack_pixel_data(image)?;
        let data_size = pixel_bytes.len() as u32;
        output.extend_from_slice(&pixel_bytes);

        // Align to word boundary
        if output.len() % 2 != 0 {
            output.push(0);
        }

        // Write IFD
        let ifd_offset = output.len() as u32;
        // Update IFD offset in header
        output[ifd_offset_pos..ifd_offset_pos + 4].copy_from_slice(&ifd_offset.to_le_bytes());

        Self::write_ifd(&mut output, image, data_offset, data_size)?;

        Ok(output)
    }

    /// Write a DNG from RGB data stored as a linear raw DNG.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimensions or data are invalid.
    pub fn write_from_rgb(
        data: &[u16],
        width: u32,
        height: u32,
        bit_depth: u8,
        metadata: &DngMetadata,
    ) -> ImageResult<Vec<u8>> {
        let expected_len = width as usize * height as usize * 3;
        if data.len() < expected_len {
            return Err(ImageError::invalid_format(format!(
                "RGB data length {} is less than expected {} ({}x{}x3)",
                data.len(),
                expected_len,
                width,
                height
            )));
        }

        let image = DngImage {
            width,
            height,
            bit_depth,
            channels: 3,
            raw_data: data.to_vec(),
            metadata: metadata.clone(),
            is_demosaiced: true,
        };

        Self::write(&image)
    }

    fn pack_pixel_data(image: &DngImage) -> ImageResult<Vec<u8>> {
        // Always write as 16-bit uncompressed for simplicity and losslessness
        let mut bytes = Vec::with_capacity(image.raw_data.len() * 2);
        for &val in &image.raw_data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        Ok(bytes)
    }

    /// Writes the IFD (Image File Directory) and its out-of-line value data.
    ///
    /// On-disk layout from `output.len()` at entry to this function onward:
    ///
    /// ```text
    /// [ 2 bytes  ] entry count
    /// [ 12 bytes ] * count   IFD entries: tag(2) type(2) count(4) value/offset(4)
    /// [ 4 bytes  ] next-IFD offset (0: this writer emits a single IFD, no SubIFD)
    /// [ N bytes  ] deferred value data, one block per entry whose value does
    ///              not fit in its 12-byte entry (type_size * count > 4)
    /// ```
    ///
    /// Entries are required by the TIFF spec to be sorted in ascending tag
    /// order. Deferred value blocks are laid out in that same (sorted) order
    /// and each entry's value/offset field is set to its block's absolute
    /// file offset, computed and written in a single pass so the two can
    /// never drift apart. Each deferred block is padded to an even length so
    /// the next block also starts on a word (even) boundary, per the TIFF
    /// spec's alignment requirement for out-of-line values.
    fn write_ifd(
        output: &mut Vec<u8>,
        image: &DngImage,
        data_offset: u32,
        data_size: u32,
    ) -> ImageResult<()> {
        // Collect all tag entries we want to write.
        let mut tags: Vec<PendingTag> = Vec::new();

        // ImageWidth
        tags.push(PendingTag::inline(TAG_IMAGE_WIDTH, 4, 1, image.width)); // LONG
                                                                           // ImageLength
        tags.push(PendingTag::inline(TAG_IMAGE_LENGTH, 4, 1, image.height));
        // BitsPerSample (always 16 since we pack to 16-bit)
        tags.push(PendingTag::inline(TAG_BITS_PER_SAMPLE, 3, 1, 16)); // SHORT
                                                                      // Compression (uncompressed)
        tags.push(PendingTag::inline(
            TAG_COMPRESSION,
            3,
            1,
            DngCompression::Uncompressed.to_u16() as u32,
        ));
        // PhotometricInterpretation
        let photometric: u32 = if image.channels == 1 {
            32803 // CFA
        } else {
            2 // RGB (for linear raw or demosaiced)
        };
        tags.push(PendingTag::inline(
            TAG_PHOTOMETRIC_INTERPRETATION,
            3,
            1,
            photometric,
        ));
        // StripOffsets (points into the pixel-data region written before the
        // IFD; this offset is computed directly by the caller, not part of
        // the deferred-data area below).
        tags.push(PendingTag::inline(TAG_STRIP_OFFSETS, 4, 1, data_offset));
        // SamplesPerPixel
        tags.push(PendingTag::inline(
            TAG_SAMPLES_PER_PIXEL,
            3,
            1,
            u32::from(image.channels),
        ));
        // RowsPerStrip
        tags.push(PendingTag::inline(TAG_ROWS_PER_STRIP, 4, 1, image.height));
        // StripByteCounts
        tags.push(PendingTag::inline(TAG_STRIP_BYTE_COUNTS, 4, 1, data_size));

        // DNG Version tag (4 BYTE values, fits inline)
        let dng_ver = &image.metadata.dng_version;
        let dng_ver_u32 = u32::from_le_bytes([dng_ver[0], dng_ver[1], dng_ver[2], dng_ver[3]]);
        tags.push(PendingTag::inline(TAG_DNG_VERSION, 1, 4, dng_ver_u32)); // BYTE

        // DNG Backward Version
        tags.push(PendingTag::inline(
            TAG_DNG_BACKWARD_VERSION,
            1,
            4,
            u32::from_le_bytes([1, 1, 0, 0]),
        ));

        // CFA pattern (only for single-channel CFA data)
        if image.channels == 1 {
            // CFA Repeat Pattern Dim (2x2, two SHORTs packed into 4 bytes)
            let dim_val = 2u32 | (2u32 << 16);
            tags.push(PendingTag::inline(
                TAG_CFA_REPEAT_PATTERN_DIM,
                3,
                2,
                dim_val,
            ));

            // CFA Pattern (4 bytes, fits inline)
            let cfa_bytes = image.metadata.cfa_pattern.as_bytes();
            let cfa_u32 = u32::from_le_bytes(cfa_bytes);
            tags.push(PendingTag::inline(TAG_CFA_PATTERN, 1, 4, cfa_u32));
        }

        // Camera model (deferred if the null-terminated string is > 4 bytes)
        if !image.metadata.camera_model.is_empty() {
            let model_bytes: Vec<u8> = image
                .metadata
                .camera_model
                .as_bytes()
                .iter()
                .copied()
                .chain(std::iter::once(0u8)) // null terminator
                .collect();
            let count = model_bytes.len() as u32;

            if count <= 4 {
                let mut inline = [0u8; 4];
                for (i, &b) in model_bytes.iter().enumerate().take(4) {
                    inline[i] = b;
                }
                tags.push(PendingTag::inline(
                    TAG_UNIQUE_CAMERA_MODEL,
                    2,
                    count,
                    u32::from_le_bytes(inline),
                ));
            } else {
                tags.push(PendingTag::deferred(
                    TAG_UNIQUE_CAMERA_MODEL,
                    2,
                    count,
                    model_bytes,
                ));
            }
        }

        // Software tag (always deferred: "OxiMedia DNG\0" is 13 bytes)
        {
            let sw = b"OxiMedia DNG\0";
            let count = sw.len() as u32;
            tags.push(PendingTag::deferred(TAG_SOFTWARE, 2, count, sw.to_vec()));
        }

        // Sort tags by tag number (TIFF requirement: IFD entries must be in
        // ascending tag order).
        tags.sort_by_key(|t| t.tag);

        let tag_count = tags.len() as u16;

        // The deferred data area begins immediately after the IFD itself:
        // the 2-byte count, `tag_count` 12-byte entries, and the 4-byte
        // next-IFD offset.
        let ifd_start = output.len();
        let ifd_entries_size = 2 + (tags.len() * 12) + 4; // count + entries + next_ifd_offset
        let mut deferred_cursor = (ifd_start + ifd_entries_size) as u32;

        // Resolve every entry's on-disk 4-byte value (inline value, or the
        // offset of its payload in the deferred area) and build the
        // deferred data area in the SAME pass, walking `tags` in the SAME
        // (already-sorted) order the entries will be emitted in. Because the
        // offset for a deferred entry is read off `deferred_cursor` at the
        // exact moment its bytes are appended to `deferred_data`, the offset
        // is guaranteed to match the byte's real position -- there is no
        // second, independently-ordered pass that could drift out of sync.
        let mut deferred_data: Vec<u8> = Vec::new();
        let mut resolved: Vec<(u16, u16, u32, u32)> = Vec::with_capacity(tags.len());
        for entry in &tags {
            let value = match &entry.value {
                TagValue::Inline(v) => *v,
                TagValue::Deferred(bytes) => {
                    let offset = deferred_cursor;
                    deferred_data.extend_from_slice(bytes);
                    deferred_cursor += bytes.len() as u32;
                    // Pad to an even offset so the next deferred value also
                    // starts word-aligned, per the TIFF spec.
                    if bytes.len() % 2 != 0 {
                        deferred_data.push(0);
                        deferred_cursor += 1;
                    }
                    offset
                }
            };
            resolved.push((entry.tag, entry.dtype, entry.count, value));
        }

        // Write IFD entry count
        output.extend_from_slice(&tag_count.to_le_bytes());

        // Write tag entries
        for &(tag, dtype, count, value) in &resolved {
            output.extend_from_slice(&tag.to_le_bytes());
            output.extend_from_slice(&dtype.to_le_bytes());
            output.extend_from_slice(&count.to_le_bytes());
            output.extend_from_slice(&value.to_le_bytes());
        }

        // Next IFD offset (0 = none; this writer emits a single IFD)
        output.extend_from_slice(&0u32.to_le_bytes());

        // Write deferred data (already laid out in the same order used to
        // compute the offsets above).
        output.extend_from_slice(&deferred_data);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dng::parser::{ByteOrder, TiffParser};
    use crate::dng::reader::DngReader;

    /// Builds a small single-channel (CFA) DNG image. `camera_model` is
    /// deliberately configurable: any value of 4+ characters forces
    /// `UniqueCameraModel` into the deferred IFD data area alongside the
    /// always-deferred `Software` tag, which is exactly the path the offset
    /// bug corrupted (see `write_ifd`'s doc comment).
    fn sample_image(camera_model: &str) -> DngImage {
        let width = 6u32;
        let height = 4u32;
        let pixel_count = (width * height) as usize;
        let raw_data: Vec<u16> = (0..pixel_count as u32)
            .map(|i| ((i * 97 + 5) % 65536) as u16)
            .collect();

        let metadata = DngMetadata {
            camera_model: camera_model.to_string(),
            ..DngMetadata::default()
        };

        DngImage {
            width,
            height,
            bit_depth: 16,
            channels: 1,
            raw_data,
            metadata,
            is_demosaiced: false,
        }
    }

    /// Round-trips a DNG through a temp file using this crate's own writer
    /// and reader.
    ///
    /// Before the offset fix, `UniqueCameraModel` (tag 50708) and `Software`
    /// (tag 305) were both deferred, but their offsets were computed by
    /// walking the *sorted* tag list while their bytes were appended to the
    /// deferred buffer in *push* order. Because Software (305) sorts before
    /// UniqueCameraModel (50708) despite being pushed after it, the two
    /// tags' offsets pointed at the wrong bytes (or past the end of the
    /// file), so `camera_model` came back empty or truncated instead of
    /// round-tripping. This test exercises exactly that path end-to-end
    /// through the crate's own parser.
    #[test]
    fn dng_writer_round_trip_via_tempfile() {
        let image = sample_image("OxiMedia Test Camera 9000");
        let bytes = DngWriter::write(&image).expect("DNG write must succeed");

        let mut path = std::env::temp_dir();
        path.push(format!(
            "oximedia_dng_writer_roundtrip_{}.dng",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("write temp DNG");
        let read_back_bytes = std::fs::read(&path).expect("read temp DNG");
        let _ = std::fs::remove_file(&path);

        assert!(
            DngReader::is_dng(&read_back_bytes),
            "writer output must be recognized as a valid DNG"
        );

        let decoded = DngReader::read(&read_back_bytes).expect("DNG round-trip read must succeed");

        assert_eq!(decoded.width, image.width);
        assert_eq!(decoded.height, image.height);
        assert_eq!(decoded.channels, image.channels);
        assert_eq!(decoded.bit_depth, 16);
        assert_eq!(decoded.metadata.dng_version, image.metadata.dng_version);
        assert_eq!(decoded.metadata.cfa_pattern, image.metadata.cfa_pattern);
        // Proves UniqueCameraModel's deferred-data offset is correct: this
        // string only lives past the IFD, at whatever offset the writer
        // computed for it.
        assert_eq!(decoded.metadata.camera_model, "OxiMedia Test Camera 9000");
        assert_eq!(decoded.raw_data, image.raw_data);
    }

    /// Same as above but with an empty camera model, so `Software` is the
    /// *only* deferred tag. A single deferred item can't collide with
    /// another one's bytes, so this path was already correct before the
    /// fix; this test guards against the fix regressing it.
    #[test]
    fn dng_writer_round_trip_no_camera_model() {
        let image = sample_image("");
        let bytes = DngWriter::write(&image).expect("DNG write must succeed");
        let decoded = DngReader::read(&bytes).expect("DNG read must succeed");

        assert_eq!(decoded.width, image.width);
        assert_eq!(decoded.height, image.height);
        assert_eq!(decoded.metadata.camera_model, "");
        assert_eq!(decoded.raw_data, image.raw_data);
    }

    /// Byte-level proof that every IFD entry whose value does not fit inline
    /// (`type_size * count > 4`) carries an offset that (a) lies within the
    /// file and (b) points at exactly the bytes that entry is supposed to
    /// hold.
    ///
    /// This inspects the raw IFD produced by the writer directly (via the
    /// crate's own `TiffParser`) rather than going through the metadata
    /// parser, so it cannot be fooled by a reader that happens to tolerate
    /// bad offsets -- it is the most direct possible check on the bug.
    #[test]
    fn dng_writer_deferred_offsets_point_at_correct_bytes() {
        let camera_model = "Deferred Offset Probe Camera";
        let image = sample_image(camera_model);
        let bytes = DngWriter::write(&image).expect("DNG write must succeed");

        let (byte_order, ifds) = TiffParser::parse(&bytes).expect("parse written DNG");
        assert_eq!(byte_order, ByteOrder::LittleEndian);
        let ifd = ifds.first().expect("writer must emit at least one IFD");

        let mut expected_model_bytes: Vec<u8> = camera_model.as_bytes().to_vec();
        expected_model_bytes.push(0); // null terminator written by the writer
        let expected_software_bytes: Vec<u8> = b"OxiMedia DNG\0".to_vec();

        let mut checked_deferred_entries = 0usize;
        for entry in &ifd.entries {
            let type_size = tiff_type_size(entry.data_type);
            let total_size = type_size * entry.count as usize;
            if total_size <= 4 {
                continue; // inline value; no offset to validate
            }

            let offset = entry.value_offset as usize;
            assert!(
                offset + total_size <= bytes.len(),
                "tag {} offset {offset} + size {total_size} exceeds file length {}",
                entry.tag,
                bytes.len()
            );
            let actual = &bytes[offset..offset + total_size];

            if entry.tag == TAG_UNIQUE_CAMERA_MODEL {
                assert_eq!(
                    actual,
                    expected_model_bytes.as_slice(),
                    "UniqueCameraModel offset does not point at the camera model string"
                );
                checked_deferred_entries += 1;
            } else if entry.tag == TAG_SOFTWARE {
                assert_eq!(
                    actual,
                    expected_software_bytes.as_slice(),
                    "Software offset does not point at the software string"
                );
                checked_deferred_entries += 1;
            }
        }

        // Both deferred tags (UniqueCameraModel and Software) must have been
        // present and validated; if this were 0 the test would trivially
        // pass without proving anything.
        assert_eq!(
            checked_deferred_entries, 2,
            "expected both UniqueCameraModel and Software to be deferred and checked"
        );
    }

    /// Every deferred value must start on an even (word-aligned) file
    /// offset, per the TIFF spec's alignment requirement for out-of-line
    /// value data.
    #[test]
    fn dng_writer_deferred_offsets_are_word_aligned() {
        let image = sample_image("Word Alignment Probe Camera");
        let bytes = DngWriter::write(&image).expect("DNG write must succeed");
        let (_, ifds) = TiffParser::parse(&bytes).expect("parse written DNG");
        let ifd = ifds.first().expect("writer must emit at least one IFD");

        let mut saw_deferred_entry = false;
        for entry in &ifd.entries {
            let total_size = tiff_type_size(entry.data_type) * entry.count as usize;
            if total_size > 4 {
                saw_deferred_entry = true;
                assert_eq!(
                    entry.value_offset % 2,
                    0,
                    "tag {} deferred offset {} is not word-aligned",
                    entry.tag,
                    entry.value_offset
                );
            }
        }
        assert!(
            saw_deferred_entry,
            "expected at least one deferred entry in this fixture"
        );
    }
}
