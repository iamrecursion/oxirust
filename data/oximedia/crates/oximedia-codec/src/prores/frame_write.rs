//! ProRes frame container and picture/slice table writer.
//!
//! This is the inverse of [`super::frame::parse_frame_header`] and
//! [`super::picture::parse_picture_header`]. The output is a complete
//! `'icpf'` frame that the decoder can parse byte-for-byte.
//!
//! ## Frame layout
//!
//! ```text
//! [0..3]   frame_size (4 bytes BE) — total including this field
//! [4..7]   'icpf' tag
//! [8..]    frame payload:
//!              frame header (20 bytes, default matrices)
//!              picture header (8 bytes)
//!              slice offset table (2 bytes × slice_count)
//!              slice 0 … slice N-1
//! ```
//!
//! ## Frame header (20 bytes, no custom matrices)
//!
//! ```text
//! [0..1]   header_size = 20 (u16 BE)
//! [2]      version = 0 (bits 7..4), reserved 0 (bits 3..0)
//! [3..6]   encoder_identifier (4-byte FourCC per profile)
//! [7..8]   width (u16 BE)
//! [9..10]  height (u16 BE)
//! [11]     chroma_format << 6 | interlace_mode << 2 | 0
//! [12]     aspect_ratio_code << 4 | frame_rate_code
//! [13]     color_primaries
//! [14]     transfer_characteristic
//! [15]     matrix_coefficients
//! [16]     source_pixel_format << 4 | alpha_channel_type
//! [17]     reserved = 0
//! [18]     quant_flags = 0 (no custom matrices)
//! [19]     reserved = 0
//! ```
//!
//! ## Picture header (8 bytes, no slice offset table in-band)
//!
//! ```text
//! [0]      header_size = 8
//! [1..4]   picture_size (u32 BE) = 8 + slice_offset_table_size + sum(slice sizes)
//! [5..6]   slice_count (u16 BE)
//! [7]      log2_slice_mb_width << 4 | 0
//! ```
//!
//! After the picture header, the slice-size table:
//! ```text
//! [0..1] slice_0_size (u16 BE)
//! [2..3] slice_1_size (u16 BE)
//! ...
//! ```
//! Then the slice data concatenated.

use super::frame::{ChromaFormat, InterlaceMode, ProResProfile};

/// Write a complete ProRes 422 `'icpf'` frame.
///
/// `slices` is the ordered list of already-encoded slice byte vectors.
/// Each `Vec<u8>` must already include the slice header and all plane
/// payloads (i.e. the output of [`super::encode::encode_slice`]).
///
/// `frame_rate_code` should match RDD 36 Table 6 (e.g. `0x06` = 25 fps,
/// `0x03` = 29.97 fps, `0x05` = 30 fps). Use `0x00` for unspecified.
///
/// `log2_slice_mb_width` is the exponent such that `1 << log2_slice_mb_width`
/// gives the number of macroblocks per slice row (default 3 = 8 MBs).
#[must_use]
pub fn write_frame(
    slices: &[Vec<u8>],
    profile: ProResProfile,
    width: u16,
    height: u16,
    frame_rate_code: u8,
    chroma: ChromaFormat,
    interlace: InterlaceMode,
    scan_order: u8,
    log2_slice_mb_width: u8,
) -> Vec<u8> {
    let fourcc = profile_fourcc(profile);

    let chroma_code: u8 = match chroma {
        ChromaFormat::Yuv422 => 2,
        ChromaFormat::Yuv444 => 3,
    };
    let interlace_code: u8 = match interlace {
        InterlaceMode::Progressive => 0,
        InterlaceMode::TopFieldFirst => 1,
        InterlaceMode::BottomFieldFirst => 2,
    };

    // ── Frame header (20 bytes) ───────────────────────────────────────
    let mut frame_hdr = Vec::with_capacity(20);
    frame_hdr.extend_from_slice(&20u16.to_be_bytes()); // [0..1] header_size
    frame_hdr.push(0x00); // [2]   version=0
    frame_hdr.extend_from_slice(&fourcc); // [3..6] encoder_id
    frame_hdr.extend_from_slice(&width.to_be_bytes()); // [7..8]
    frame_hdr.extend_from_slice(&height.to_be_bytes()); // [9..10]
    frame_hdr.push((chroma_code << 6) | (interlace_code << 2)); // [11]
    frame_hdr.push((scan_order << 4) | (frame_rate_code & 0x0F)); // [12]
    frame_hdr.push(1); // [13] color_primaries = BT.709
    frame_hdr.push(1); // [14] transfer_characteristic = BT.709
    frame_hdr.push(1); // [15] matrix_coefficients = BT.709
    frame_hdr.push(0x00); // [16] source_pixel_format=0, alpha_channel_type=0
    frame_hdr.push(0x00); // [17] reserved
    frame_hdr.push(0x00); // [18] no custom quant matrices
    frame_hdr.push(0x00); // [19] reserved
    debug_assert_eq!(frame_hdr.len(), 20);

    // ── Slice offset table (2 bytes per slice) ───────────────────────
    let slice_count = slices.len() as u16;
    let offset_table_size = 2 * slice_count as usize;
    let mut offset_table = Vec::with_capacity(offset_table_size);
    for slice in slices {
        let sz = slice.len() as u16;
        offset_table.extend_from_slice(&sz.to_be_bytes());
    }

    // ── Slice data (concatenated) ─────────────────────────────────────
    let slice_data_size: usize = slices.iter().map(|s| s.len()).sum();

    // ── Picture header (8 bytes) ─────────────────────────────────────
    // picture_size includes the picture header itself + offset table + slice data.
    let picture_size = 8u32 + offset_table_size as u32 + slice_data_size as u32;
    let mut pic_hdr = Vec::with_capacity(8);
    pic_hdr.push(8u8); // [0] header_size = 8
    pic_hdr.extend_from_slice(&picture_size.to_be_bytes()); // [1..4]
    pic_hdr.extend_from_slice(&slice_count.to_be_bytes()); // [5..6]
    pic_hdr.push(log2_slice_mb_width << 4); // [7]
    debug_assert_eq!(pic_hdr.len(), 8);

    // ── Assemble payload ─────────────────────────────────────────────
    // payload = frame_hdr + pic_hdr + offset_table + slice_data
    let payload_size = frame_hdr.len() + pic_hdr.len() + offset_table_size + slice_data_size;
    let frame_size = 8u32 + payload_size as u32; // 8 = size(4) + 'icpf'(4)

    let mut out = Vec::with_capacity(frame_size as usize);
    // Container header.
    out.extend_from_slice(&frame_size.to_be_bytes());
    out.extend_from_slice(b"icpf");
    // Frame header.
    out.extend_from_slice(&frame_hdr);
    // Picture header.
    out.extend_from_slice(&pic_hdr);
    // Slice offset table.
    out.extend_from_slice(&offset_table);
    // Slice data.
    for slice in slices {
        out.extend_from_slice(slice);
    }

    out
}

/// Return the 4-byte FourCC for the given profile, matching the decoder's
/// `ProResProfile::from_fourcc`.
fn profile_fourcc(profile: ProResProfile) -> [u8; 4] {
    match profile {
        ProResProfile::Proxy => *b"apco",
        ProResProfile::Lt => *b"apcs",
        ProResProfile::Standard => *b"apcn",
        ProResProfile::Hq => *b"apch",
        ProResProfile::P4444 => *b"ap4h",
        ProResProfile::P4444Xq => *b"ap4x",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prores::frame::{parse_frame_header, FrameContainer};
    use crate::prores::picture::parse_picture_header;

    fn dummy_slices(n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| vec![0u8; 20 + i]).collect()
    }

    #[test]
    fn write_frame_round_trips_through_parser() {
        let slices = dummy_slices(4);
        let frame = write_frame(
            &slices,
            ProResProfile::Standard,
            32,
            16,
            0x05, // 30fps
            ChromaFormat::Yuv422,
            InterlaceMode::Progressive,
            0,
            3,
        );

        // Parse container.
        let (container, rest) = FrameContainer::parse(&frame).expect("container parse");
        assert!(rest.is_empty(), "no trailing bytes expected");

        // Parse frame header.
        let (fhdr, after_fhdr) = parse_frame_header(container.payload).expect("frame header");
        assert_eq!(fhdr.width, 32);
        assert_eq!(fhdr.height, 16);
        assert_eq!(fhdr.profile, ProResProfile::Standard);
        assert_eq!(fhdr.chroma_format, ChromaFormat::Yuv422);
        assert_eq!(fhdr.interlace_mode, InterlaceMode::Progressive);

        // Parse picture header.
        let (pic_hdr, after_pic) = parse_picture_header(after_fhdr).expect("picture header");
        assert_eq!(pic_hdr.slice_count, 4);
        assert_eq!(pic_hdr.log2_slice_mb_width, 3);

        // Slice offset table: 4 slices × 2 bytes = 8 bytes.
        let offset_table = &after_pic[..8];
        for (i, chunk) in offset_table.chunks(2).enumerate() {
            let sz = u16::from_be_bytes([chunk[0], chunk[1]]) as usize;
            assert_eq!(sz, slices[i].len(), "slice {i} size mismatch in table");
        }
    }

    #[test]
    fn write_frame_frame_size_field_is_correct() {
        let slices = vec![vec![0xABu8; 100]];
        let frame = write_frame(
            &slices,
            ProResProfile::Hq,
            1920,
            1080,
            0x05,
            ChromaFormat::Yuv422,
            InterlaceMode::Progressive,
            0,
            3,
        );
        let declared_size = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        assert_eq!(
            declared_size,
            frame.len(),
            "frame_size field should match actual length"
        );
    }

    #[test]
    fn write_frame_all_profiles() {
        for profile in [
            ProResProfile::Proxy,
            ProResProfile::Lt,
            ProResProfile::Standard,
            ProResProfile::Hq,
        ] {
            let frame = write_frame(
                &dummy_slices(2),
                profile,
                16,
                16,
                0,
                ChromaFormat::Yuv422,
                InterlaceMode::Progressive,
                0,
                3,
            );
            let (container, _) = FrameContainer::parse(&frame).expect("container");
            let (fhdr, _) = parse_frame_header(container.payload).expect("frame hdr");
            assert_eq!(fhdr.profile, profile);
        }
    }
}
