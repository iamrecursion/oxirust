//! Tests for DNG module.

use std::collections::HashMap;

use crate::dng::constants::*;
use crate::dng::conversion::{dng_to_image_frame, image_frame_to_dng};
use crate::dng::demosaic::demosaic_bilinear;
use crate::dng::parser::{ByteOrder, TiffParser};
use crate::dng::processing::{apply_color_matrix, apply_white_balance};
use crate::dng::reader::DngReader;
use crate::dng::types::*;
use crate::dng::writer::DngWriter;
use crate::{ColorSpace, ImageData, ImageFrame, PixelType};

// Helper: build a minimal valid DNG in memory (little-endian)
fn build_minimal_dng(width: u32, height: u32, bps: u16, pattern: CfaPattern) -> Vec<u8> {
    let mut buf = Vec::new();

    // TIFF header
    buf.extend_from_slice(&[0x49, 0x49]); // little-endian
    buf.extend_from_slice(&42u16.to_le_bytes());
    // IFD offset placeholder (will fill after pixel data)
    let ifd_offset_pos = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes());

    // Pixel data: fill with a known pattern
    let data_offset = buf.len() as u32;
    let pixel_count = width as usize * height as usize;
    let data_size = pixel_count * 2; // 16-bit
    for i in 0..pixel_count {
        buf.extend_from_slice(&(i as u16).to_le_bytes());
    }

    // Align
    if buf.len() % 2 != 0 {
        buf.push(0);
    }

    let ifd_offset = buf.len() as u32;
    buf[ifd_offset_pos..ifd_offset_pos + 4].copy_from_slice(&ifd_offset.to_le_bytes());

    // Build IFD entries (sorted by tag)
    let mut tags: Vec<(u16, u16, u32, u32)> = vec![
        (TAG_IMAGE_WIDTH, 4, 1, width),
        (TAG_IMAGE_LENGTH, 4, 1, height),
        (TAG_BITS_PER_SAMPLE, 3, 1, u32::from(bps)),
        (TAG_COMPRESSION, 3, 1, 1),                    // uncompressed
        (TAG_PHOTOMETRIC_INTERPRETATION, 3, 1, 32803), // CFA
        (TAG_STRIP_OFFSETS, 4, 1, data_offset),
        (TAG_SAMPLES_PER_PIXEL, 3, 1, 1),
        (TAG_ROWS_PER_STRIP, 4, 1, height),
        (TAG_STRIP_BYTE_COUNTS, 4, 1, data_size as u32),
        (TAG_CFA_REPEAT_PATTERN_DIM, 3, 2, 2 | (2 << 16)),
        (
            TAG_CFA_PATTERN,
            1,
            4,
            u32::from_le_bytes(pattern.as_bytes()),
        ),
        // DNG Version [1, 4, 0, 0]
        (TAG_DNG_VERSION, 1, 4, u32::from_le_bytes([1, 4, 0, 0])),
    ];
    tags.sort_by_key(|t| t.0);

    let tag_count = tags.len() as u16;
    buf.extend_from_slice(&tag_count.to_le_bytes());

    for &(tag, dtype, count, value) in &tags {
        buf.extend_from_slice(&tag.to_le_bytes());
        buf.extend_from_slice(&dtype.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }

    // Next IFD = 0
    buf.extend_from_slice(&0u32.to_le_bytes());

    buf
}

fn build_minimal_dng_be(width: u32, height: u32) -> Vec<u8> {
    let mut buf = Vec::new();

    // TIFF header (big-endian)
    buf.extend_from_slice(&[0x4D, 0x4D]); // "MM"
    buf.extend_from_slice(&42u16.to_be_bytes());
    let ifd_offset_pos = buf.len();
    buf.extend_from_slice(&0u32.to_be_bytes());

    // Pixel data
    let data_offset = buf.len() as u32;
    let pixel_count = width as usize * height as usize;
    let data_size = pixel_count * 2;
    for i in 0..pixel_count {
        buf.extend_from_slice(&(i as u16).to_be_bytes());
    }

    if buf.len() % 2 != 0 {
        buf.push(0);
    }

    let ifd_offset = buf.len() as u32;
    buf[ifd_offset_pos..ifd_offset_pos + 4].copy_from_slice(&ifd_offset.to_be_bytes());

    let mut tags: Vec<(u16, u16, u32, u32)> = vec![
        (TAG_IMAGE_WIDTH, 4, 1, width),
        (TAG_IMAGE_LENGTH, 4, 1, height),
        (TAG_BITS_PER_SAMPLE, 3, 1, 16),
        (TAG_COMPRESSION, 3, 1, 1),
        (TAG_PHOTOMETRIC_INTERPRETATION, 3, 1, 32803),
        (TAG_STRIP_OFFSETS, 4, 1, data_offset),
        (TAG_SAMPLES_PER_PIXEL, 3, 1, 1),
        (TAG_ROWS_PER_STRIP, 4, 1, height),
        (TAG_STRIP_BYTE_COUNTS, 4, 1, data_size as u32),
        (TAG_CFA_PATTERN, 1, 4, u32::from_be_bytes([0, 1, 1, 2])), // RGGB in BE
        (TAG_DNG_VERSION, 1, 4, u32::from_be_bytes([1, 4, 0, 0])),
    ];
    tags.sort_by_key(|t| t.0);

    let tag_count = tags.len() as u16;
    buf.extend_from_slice(&tag_count.to_be_bytes());

    for &(tag, dtype, count, value) in &tags {
        buf.extend_from_slice(&tag.to_be_bytes());
        buf.extend_from_slice(&dtype.to_be_bytes());
        buf.extend_from_slice(&count.to_be_bytes());
        buf.extend_from_slice(&value.to_be_bytes());
    }

    buf.extend_from_slice(&0u32.to_be_bytes());

    buf
}

#[test]
#[ignore]
fn test_tiff_header_parsing_le() {
    let data = build_minimal_dng(4, 4, 16, CfaPattern::Rggb);
    let result = TiffParser::parse(&data);
    assert!(result.is_ok());
    let (byte_order, ifds) = result.expect("parse failed");
    assert_eq!(byte_order, ByteOrder::LittleEndian);
    assert!(!ifds.is_empty());
}

#[test]
#[ignore]
fn test_tiff_header_parsing_be() {
    let data = build_minimal_dng_be(4, 4);
    let result = TiffParser::parse(&data);
    assert!(result.is_ok());
    let (byte_order, ifds) = result.expect("parse failed");
    assert_eq!(byte_order, ByteOrder::BigEndian);
    assert!(!ifds.is_empty());
}

#[test]
#[ignore]
fn test_ifd_parsing() {
    let data = build_minimal_dng(8, 6, 16, CfaPattern::Rggb);
    let (byte_order, ifds) = TiffParser::parse(&data).expect("parse failed");
    let parser = TiffParser { byte_order };
    let ifd = &ifds[0];

    let width = parser.get_tag_value_u32(ifd, TAG_IMAGE_WIDTH, &data);
    assert_eq!(width, Some(8));
    let height = parser.get_tag_value_u32(ifd, TAG_IMAGE_LENGTH, &data);
    assert_eq!(height, Some(6));
}

#[test]
#[ignore]
fn test_dng_detection_valid() {
    let data = build_minimal_dng(4, 4, 16, CfaPattern::Rggb);
    assert!(DngReader::is_dng(&data));
}

#[test]
#[ignore]
fn test_dng_detection_invalid_tiff() {
    // Valid TIFF but no DNG version tag
    let mut data = vec![0x49, 0x49]; // LE
    data.extend_from_slice(&42u16.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes()); // IFD at 8

    // Minimal IFD with just width
    let tag_count: u16 = 1;
    data.extend_from_slice(&tag_count.to_le_bytes());
    // ImageWidth tag
    data.extend_from_slice(&256u16.to_le_bytes()); // tag
    data.extend_from_slice(&4u16.to_le_bytes()); // LONG
    data.extend_from_slice(&1u32.to_le_bytes()); // count
    data.extend_from_slice(&10u32.to_le_bytes()); // value
                                                  // Next IFD
    data.extend_from_slice(&0u32.to_le_bytes());

    assert!(!DngReader::is_dng(&data));
}

#[test]
#[ignore]
fn test_dng_detection_garbage() {
    assert!(!DngReader::is_dng(&[0, 1, 2, 3]));
    assert!(!DngReader::is_dng(&[]));
    assert!(!DngReader::is_dng(&[0x49, 0x49, 0xFF, 0xFF]));
}

#[test]
#[ignore]
fn test_cfa_pattern_parsing() {
    assert_eq!(
        DngReader::parse_cfa_pattern(&[0, 1, 1, 2]).expect("parse"),
        CfaPattern::Rggb
    );
    assert_eq!(
        DngReader::parse_cfa_pattern(&[2, 1, 1, 0]).expect("parse"),
        CfaPattern::Bggr
    );
    assert_eq!(
        DngReader::parse_cfa_pattern(&[1, 0, 2, 1]).expect("parse"),
        CfaPattern::Grbg
    );
    assert_eq!(
        DngReader::parse_cfa_pattern(&[1, 2, 0, 1]).expect("parse"),
        CfaPattern::Gbrg
    );

    // Invalid pattern
    assert!(DngReader::parse_cfa_pattern(&[3, 3, 3, 3]).is_err());
    // Too short
    assert!(DngReader::parse_cfa_pattern(&[0, 1]).is_err());
}

#[test]
#[ignore]
fn test_bit_unpacking_8bit() {
    let data = vec![10, 20, 30, 40];
    let result = DngReader::unpack_bits(&data, 8, 4).expect("unpack");
    assert_eq!(result, vec![10, 20, 30, 40]);
}

#[test]
#[ignore]
fn test_bit_unpacking_16bit() {
    let data: Vec<u8> = vec![0x00, 0x04, 0xFF, 0x0F]; // 1024, 4095
    let result = DngReader::unpack_bits(&data, 16, 2).expect("unpack");
    assert_eq!(result, vec![1024, 4095]);
}

#[test]
#[ignore]
fn test_bit_unpacking_10bit() {
    // 10-bit: pack 4 values into 5 bytes
    // Values: 1023 (0x3FF), 512 (0x200), 0 (0x000), 1 (0x001)
    // Bit stream: 11_1111_1111 10_0000_0000 00_0000_0000 00_0000_0001
    // = 0xFF 0xE8 0x00 0x00 0x40 (40 bits = 5 bytes)
    let bits: u64 = (1023u64 << 30) | (512u64 << 20) | (0u64 << 10) | 1u64;
    let bytes = [
        ((bits >> 32) & 0xFF) as u8,
        ((bits >> 24) & 0xFF) as u8,
        ((bits >> 16) & 0xFF) as u8,
        ((bits >> 8) & 0xFF) as u8,
        (bits & 0xFF) as u8,
    ];

    let result = DngReader::unpack_bits(&bytes, 10, 4).expect("unpack");
    assert_eq!(result, vec![1023, 512, 0, 1]);
}

#[test]
#[ignore]
fn test_bit_unpacking_12bit() {
    // 12-bit: pack 2 values into 3 bytes
    // Values: 4095 (0xFFF), 2048 (0x800)
    // Bit stream: 1111_1111_1111 1000_0000_0000
    // = 0xFF 0xF8 0x00 (24 bits = 3 bytes)
    let bits: u32 = (4095u32 << 12) | 2048u32;
    let bytes = [
        ((bits >> 16) & 0xFF) as u8,
        ((bits >> 8) & 0xFF) as u8,
        (bits & 0xFF) as u8,
    ];

    let result = DngReader::unpack_bits(&bytes, 12, 2).expect("unpack");
    assert_eq!(result, vec![4095, 2048]);
}

#[test]
#[ignore]
fn test_bit_unpacking_14bit() {
    // 14-bit: pack 2 values into 28 bits (4 bytes with padding)
    // Values: 16383 (0x3FFF), 8192 (0x2000)
    // Bit stream (MSB first): 11_1111_1111_1111 10_0000_0000_0000 0000
    // 28 bits of real data + 4 bits padding = 32 bits = 4 bytes
    let bits: u64 = (16383u64 << 14) | 8192u64;
    // Shift left by 4 to align to MSB of 32 bits (32 - 28 = 4)
    let shifted = bits << 4;
    let bytes = [
        ((shifted >> 24) & 0xFF) as u8,
        ((shifted >> 16) & 0xFF) as u8,
        ((shifted >> 8) & 0xFF) as u8,
        (shifted & 0xFF) as u8,
    ];

    let result = DngReader::unpack_bits(&bytes, 14, 2).expect("unpack");
    assert_eq!(result, vec![16383, 8192]);
}

#[test]
#[ignore]
fn test_demosaic_bilinear_rggb() {
    // Create a 4x4 synthetic Bayer RGGB pattern
    // Pattern:
    //   R  G  R  G
    //   G  B  G  B
    //   R  G  R  G
    //   G  B  G  B
    //
    // Fill: R=1000, G=500, B=200
    let w = 4u32;
    let h = 4u32;
    let mut raw = vec![0u16; 16];

    for y in 0..4usize {
        for x in 0..4usize {
            let px = y % 2;
            let py = x % 2;
            raw[y * 4 + x] = match (px, py) {
                (0, 0) => 1000, // R
                (0, 1) => 500,  // G
                (1, 0) => 500,  // G
                (1, 1) => 200,  // B
                _ => 0,
            };
        }
    }

    let result = demosaic_bilinear(&raw, w, h, CfaPattern::Rggb).expect("demosaic");
    assert_eq!(result.len(), 16 * 3);

    // Check center pixel (1,1) which is a Blue pixel in RGGB
    // At (1,1): B=200 (known)
    // R should be interpolated from diagonals: (0,0)=1000, (0,2)=1000, (2,0)=1000, (2,2)=1000 -> 1000
    // G should be interpolated from 4-connected: (0,1)=500, (1,0)=500, (1,2)=500, (2,1)=500 -> 500
    let idx = (1 * 4 + 1) * 3;
    assert_eq!(result[idx + 2], 200, "Blue channel at (1,1)");
    assert_eq!(result[idx], 1000, "Red channel at (1,1) from diagonals");
    assert_eq!(result[idx + 1], 500, "Green channel at (1,1)");
}

#[test]
#[ignore]
fn test_demosaic_bilinear_bggr() {
    // BGGR: B G / G R
    let w = 4u32;
    let h = 4u32;
    let mut raw = vec![0u16; 16];

    for y in 0..4usize {
        for x in 0..4usize {
            let px = y % 2;
            let py = x % 2;
            raw[y * 4 + x] = match (px, py) {
                (0, 0) => 200,  // B
                (0, 1) => 500,  // G
                (1, 0) => 500,  // G
                (1, 1) => 1000, // R
                _ => 0,
            };
        }
    }

    let result = demosaic_bilinear(&raw, w, h, CfaPattern::Bggr).expect("demosaic");
    assert_eq!(result.len(), 16 * 3);

    // At (1,1): R=1000 (known)
    let idx = (1 * 4 + 1) * 3;
    assert_eq!(result[idx], 1000, "Red channel at (1,1)");
}

#[test]
#[ignore]
fn test_white_balance_application() {
    // Simple test: neutral = [0.5, 1.0, 0.5] means R and B get boosted
    let mut data = vec![100u16, 200, 100, 200, 400, 200];
    let wb = WhiteBalance {
        as_shot_neutral: [0.5, 1.0, 0.5],
    };

    apply_white_balance(&mut data, &wb, 65535);

    // Gains: R=2.0, G=1.0, B=2.0
    // Normalized by min (1.0): R=2.0, G=1.0, B=2.0
    assert_eq!(data[0], 200); // 100 * 2.0
    assert_eq!(data[1], 200); // 200 * 1.0
    assert_eq!(data[2], 200); // 100 * 2.0
}

#[test]
#[ignore]
fn test_color_matrix_identity() {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut data = vec![0.5, 0.3, 0.1, 1.0, 0.0, 0.5];

    apply_color_matrix(&mut data, &identity);

    assert!((data[0] - 0.5).abs() < 1e-10);
    assert!((data[1] - 0.3).abs() < 1e-10);
    assert!((data[2] - 0.1).abs() < 1e-10);
    assert!((data[3] - 1.0).abs() < 1e-10);
    assert!((data[4] - 0.0).abs() < 1e-10);
    assert!((data[5] - 0.5).abs() < 1e-10);
}

#[test]
#[ignore]
fn test_color_matrix_transform() {
    // Simple swap: R->B, G->R, B->G
    let matrix = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
    let mut data = vec![1.0, 2.0, 3.0];

    apply_color_matrix(&mut data, &matrix);

    assert!((data[0] - 2.0).abs() < 1e-10); // was G
    assert!((data[1] - 3.0).abs() < 1e-10); // was B
    assert!((data[2] - 1.0).abs() < 1e-10); // was R
}

#[test]
#[ignore]
fn test_round_trip_write_read() {
    let width = 8u32;
    let height = 6u32;
    let pixel_count = width as usize * height as usize;

    // Create a test DNG image with known data
    let mut raw_data = Vec::with_capacity(pixel_count);
    for i in 0..pixel_count {
        raw_data.push((i * 100 % 65536) as u16);
    }

    let metadata = DngMetadata {
        dng_version: [1, 4, 0, 0],
        camera_model: "TestCam".to_string(),
        cfa_pattern: CfaPattern::Rggb,
        white_balance: WhiteBalance::default(),
        color_calibration: ColorCalibration::default(),
        black_level: vec![0.0],
        white_level: vec![65535],
        active_area: None,
        exif: HashMap::new(),
    };

    let image = DngImage {
        width,
        height,
        bit_depth: 16,
        channels: 1,
        raw_data: raw_data.clone(),
        metadata,
        is_demosaiced: false,
    };

    // Write
    let written = DngWriter::write(&image).expect("write failed");

    // Verify it is valid DNG
    assert!(DngReader::is_dng(&written), "Written data is not valid DNG");

    // Read back
    let read_back = DngReader::read(&written).expect("read failed");

    assert_eq!(read_back.width, width);
    assert_eq!(read_back.height, height);
    assert_eq!(read_back.channels, 1);
    assert_eq!(read_back.raw_data.len(), raw_data.len());

    // Verify pixel data matches (writer always stores as 16-bit)
    for i in 0..pixel_count {
        assert_eq!(
            read_back.raw_data[i], raw_data[i],
            "Pixel mismatch at index {i}"
        );
    }

    // Verify metadata
    assert_eq!(read_back.metadata.dng_version, [1, 4, 0, 0]);
    assert_eq!(read_back.metadata.cfa_pattern, CfaPattern::Rggb);
}

#[test]
#[ignore]
fn test_dng_to_image_frame_raw() {
    let image = DngImage {
        width: 4,
        height: 4,
        bit_depth: 16,
        channels: 1,
        raw_data: vec![100u16; 16],
        metadata: DngMetadata::default(),
        is_demosaiced: false,
    };

    let frame = dng_to_image_frame(&image, false).expect("conversion failed");
    assert_eq!(frame.width, 4);
    assert_eq!(frame.height, 4);
    assert_eq!(frame.components, 1);
    assert_eq!(frame.pixel_type, PixelType::U16);
    assert_eq!(frame.color_space, ColorSpace::Luma);
}

#[test]
#[ignore]
fn test_dng_to_image_frame_demosaiced() {
    let image = DngImage {
        width: 4,
        height: 4,
        bit_depth: 16,
        channels: 1,
        raw_data: vec![500u16; 16],
        metadata: DngMetadata::default(),
        is_demosaiced: false,
    };

    let frame = dng_to_image_frame(&image, true).expect("conversion failed");
    assert_eq!(frame.width, 4);
    assert_eq!(frame.height, 4);
    assert_eq!(frame.components, 3);
    assert_eq!(frame.pixel_type, PixelType::U16);
    assert_eq!(frame.color_space, ColorSpace::LinearRgb);
}

#[test]
#[ignore]
fn test_image_frame_to_dng() {
    let byte_data: Vec<u8> = (0..48u16).flat_map(|v| v.to_le_bytes()).collect();
    let frame = ImageFrame::new(
        0,
        4,
        4,
        PixelType::U16,
        3,
        ColorSpace::LinearRgb,
        ImageData::interleaved(byte_data),
    );

    let dng = image_frame_to_dng(&frame, None).expect("conversion failed");
    assert_eq!(dng.width, 4);
    assert_eq!(dng.height, 4);
    assert_eq!(dng.channels, 3);
    assert!(dng.is_demosaiced);
    assert_eq!(dng.raw_data.len(), 48);
}

#[test]
#[ignore]
fn test_dng_read_from_constructed_data() {
    let data = build_minimal_dng(8, 8, 16, CfaPattern::Bggr);
    let image = DngReader::read(&data).expect("read failed");

    assert_eq!(image.width, 8);
    assert_eq!(image.height, 8);
    assert_eq!(image.bit_depth, 16);
    assert_eq!(image.channels, 1);
    assert_eq!(image.metadata.cfa_pattern, CfaPattern::Bggr);
    assert!(!image.is_demosaiced);
    assert_eq!(image.raw_data.len(), 64);
}

#[test]
#[ignore]
fn test_metadata_only_read() {
    let data = build_minimal_dng(16, 12, 14, CfaPattern::Grbg);
    let metadata = DngReader::read_metadata(&data).expect("read metadata failed");

    assert_eq!(metadata.dng_version, [1, 4, 0, 0]);
    assert_eq!(metadata.cfa_pattern, CfaPattern::Grbg);
}

#[test]
#[ignore]
fn test_write_from_rgb() {
    let width = 4u32;
    let height = 4u32;
    let rgb_data: Vec<u16> = (0..48).collect();
    let metadata = DngMetadata::default();

    let written =
        DngWriter::write_from_rgb(&rgb_data, width, height, 16, &metadata).expect("write");
    assert!(DngReader::is_dng(&written));

    let read_back = DngReader::read(&written).expect("read");
    assert_eq!(read_back.width, 4);
    assert_eq!(read_back.height, 4);
    assert_eq!(read_back.channels, 3);
}

#[test]
#[ignore]
fn test_cfa_pattern_color_indices() {
    assert_eq!(CfaPattern::Rggb.color_indices(), [0, 1, 1, 2]);
    assert_eq!(CfaPattern::Bggr.color_indices(), [2, 1, 1, 0]);
    assert_eq!(CfaPattern::Grbg.color_indices(), [1, 0, 2, 1]);
    assert_eq!(CfaPattern::Gbrg.color_indices(), [1, 2, 0, 1]);
}

#[test]
#[ignore]
fn test_dng_compression_conversion() {
    assert_eq!(
        DngCompression::from_u16(1).expect("parse"),
        DngCompression::Uncompressed
    );
    assert_eq!(
        DngCompression::from_u16(7).expect("parse"),
        DngCompression::LosslessJpeg
    );
    assert_eq!(
        DngCompression::from_u16(8).expect("parse"),
        DngCompression::Deflate
    );
    assert_eq!(
        DngCompression::from_u16(34892).expect("parse"),
        DngCompression::LossyDng
    );
    assert!(DngCompression::from_u16(999).is_err());
}

#[test]
#[ignore]
fn test_demosaic_edge_handling() {
    // Test that demosaicing handles 2x2 (minimum) correctly
    let raw = vec![1000u16, 500, 500, 200];
    let result = demosaic_bilinear(&raw, 2, 2, CfaPattern::Rggb).expect("demosaic");
    assert_eq!(result.len(), 12); // 4 pixels * 3 channels
                                  // No panics or out-of-bounds is the main assertion
}

#[test]
#[ignore]
fn test_white_balance_neutral_identity() {
    // Neutral [1.0, 1.0, 1.0] should not change values
    let mut data = vec![100u16, 200, 300];
    let wb = WhiteBalance {
        as_shot_neutral: [1.0, 1.0, 1.0],
    };

    apply_white_balance(&mut data, &wb, 65535);

    assert_eq!(data, vec![100, 200, 300]);
}

#[test]
#[ignore]
fn test_default_metadata() {
    let meta = DngMetadata::default();
    assert_eq!(meta.dng_version, [1, 4, 0, 0]);
    assert_eq!(meta.cfa_pattern, CfaPattern::Rggb);
    assert_eq!(meta.white_balance.as_shot_neutral, [1.0, 1.0, 1.0]);
    assert_eq!(meta.color_calibration.illuminant_1, 21);
}

// ==========================================
// Compressed-DNG decode tests (LosslessJpeg / LossyDng)
// ==========================================

/// MSB-first bit writer for synthesising lossless-JPEG entropy data.
struct LjBitWriter {
    bytes: Vec<u8>,
    current: u8,
    filled: u8,
}

impl LjBitWriter {
    fn new() -> Self {
        LjBitWriter {
            bytes: Vec::new(),
            current: 0,
            filled: 0,
        }
    }

    fn write_bit(&mut self, bit: u8) {
        self.current = (self.current << 1) | (bit & 1);
        self.filled += 1;
        if self.filled == 8 {
            self.bytes.push(self.current);
            if self.current == 0xFF {
                self.bytes.push(0x00); // JPEG byte stuffing
            }
            self.current = 0;
            self.filled = 0;
        }
    }

    fn write_bits(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            self.write_bit(((value >> i) & 1) as u8);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        while self.filled != 0 {
            self.write_bit(1);
        }
        self.bytes
    }
}

/// Magnitude category of a signed difference.
fn lj_category(diff: i32) -> u8 {
    let mut magnitude = diff.unsigned_abs();
    let mut bits = 0u8;
    while magnitude != 0 {
        magnitude >>= 1;
        bits += 1;
    }
    bits
}

/// Mantissa bits of a signed difference for the given category.
fn lj_mantissa(diff: i32, ssss: u8) -> u32 {
    if ssss == 0 {
        0
    } else if diff >= 0 {
        diff as u32
    } else {
        (diff - 1 + (1 << ssss)) as u32
    }
}

/// Lossless-JPEG predictor 1 (left); start-of-line uses the sample above,
/// start-of-image uses the default `2^(P-1)`.
fn lj_predict_p1(
    samples: &[u16],
    width: usize,
    n_comp: usize,
    x: usize,
    y: usize,
    c: usize,
) -> i32 {
    let at = |sx: usize, sy: usize| i32::from(samples[(sy * width + sx) * n_comp + c]);
    if x == 0 && y == 0 {
        1 << 15
    } else if x == 0 {
        at(0, y - 1)
    } else {
        at(x - 1, y)
    }
}

/// Build a complete lossless-JPEG (SOF3, predictor 1) datastream.
///
/// Uses a flat 17-symbol DC Huffman table (all 5-bit codes), so the canonical
/// code of category `s` is simply `s`.
fn build_lossless_jpeg_stream(width: u16, height: u16, components: u8, samples: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&[0xFF, 0xD8]); // SOI

    // SOF3
    let mut sof = Vec::new();
    sof.push(16u8); // precision
    sof.extend_from_slice(&height.to_be_bytes());
    sof.extend_from_slice(&width.to_be_bytes());
    sof.push(components);
    for c in 0..components {
        sof.push(c + 1);
        sof.push(0x11);
        sof.push(0x00);
    }
    out.extend_from_slice(&[0xFF, 0xC3]);
    out.extend_from_slice(&((sof.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&sof);

    // DHT: flat DC table — 17 codes of length 5.
    let mut dht = vec![0x00u8]; // Tc=0, Th=0
    let mut counts = [0u8; 16];
    counts[4] = 17;
    dht.extend_from_slice(&counts);
    dht.extend((0u8..=16).collect::<Vec<u8>>());
    out.extend_from_slice(&[0xFF, 0xC4]);
    out.extend_from_slice(&((dht.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&dht);

    // SOS
    let mut sos = Vec::new();
    sos.push(components);
    for c in 0..components {
        sos.push(c + 1);
        sos.push(0x00);
    }
    sos.push(0x01); // predictor 1
    sos.push(0x00); // Se
    sos.push(0x00); // Ah/Al
    out.extend_from_slice(&[0xFF, 0xDA]);
    out.extend_from_slice(&((sos.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&sos);

    // Entropy data.
    let w = width as usize;
    let h = height as usize;
    let n = components as usize;
    let mut writer = LjBitWriter::new();
    for y in 0..h {
        for x in 0..w {
            for c in 0..n {
                let actual = i32::from(samples[(y * w + x) * n + c]);
                let px = lj_predict_p1(samples, w, n, x, y, c);
                let diff = ((actual - px) & 0xFFFF) as i16 as i32;
                let ssss = lj_category(diff);
                writer.write_bits(u32::from(ssss), 5);
                if ssss > 0 {
                    writer.write_bits(lj_mantissa(diff, ssss), ssss);
                }
            }
        }
    }
    out.extend_from_slice(&writer.finish());
    out.extend_from_slice(&[0xFF, 0xD9]); // EOI
    out
}

/// Build a DNG file whose single strip carries the given pre-compressed bytes.
fn build_compressed_dng(
    width: u32,
    height: u32,
    compression: u16,
    strip_bytes: &[u8],
    pattern: CfaPattern,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&[0x49, 0x49]); // little-endian
    buf.extend_from_slice(&42u16.to_le_bytes());
    let ifd_offset_pos = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes());

    // Strip data immediately after the header.
    let data_offset = buf.len() as u32;
    buf.extend_from_slice(strip_bytes);
    if buf.len() % 2 != 0 {
        buf.push(0);
    }

    let ifd_offset = buf.len() as u32;
    buf[ifd_offset_pos..ifd_offset_pos + 4].copy_from_slice(&ifd_offset.to_le_bytes());

    let mut tags: Vec<(u16, u16, u32, u32)> = vec![
        (TAG_IMAGE_WIDTH, 4, 1, width),
        (TAG_IMAGE_LENGTH, 4, 1, height),
        (TAG_BITS_PER_SAMPLE, 3, 1, 16),
        (TAG_COMPRESSION, 3, 1, u32::from(compression)),
        (TAG_PHOTOMETRIC_INTERPRETATION, 3, 1, 32803),
        (TAG_STRIP_OFFSETS, 4, 1, data_offset),
        (TAG_SAMPLES_PER_PIXEL, 3, 1, 1),
        (TAG_ROWS_PER_STRIP, 4, 1, height),
        (TAG_STRIP_BYTE_COUNTS, 4, 1, strip_bytes.len() as u32),
        (
            TAG_CFA_PATTERN,
            1,
            4,
            u32::from_le_bytes(pattern.as_bytes()),
        ),
        (TAG_DNG_VERSION, 1, 4, u32::from_le_bytes([1, 4, 0, 0])),
    ];
    tags.sort_by_key(|t| t.0);

    buf.extend_from_slice(&(tags.len() as u16).to_le_bytes());
    for &(tag, dtype, count, value) in &tags {
        buf.extend_from_slice(&tag.to_le_bytes());
        buf.extend_from_slice(&dtype.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }
    buf.extend_from_slice(&0u32.to_le_bytes()); // next IFD
    buf
}

#[test]
fn test_dng_lossless_jpeg_single_component() {
    // 8x6 CFA stored as a single-component lossless JPEG.
    let width = 8u32;
    let height = 6u32;
    let samples: Vec<u16> = (0..(width * height))
        .map(|i| (i as u16).wrapping_mul(173).wrapping_add(500))
        .collect();
    let jpeg = build_lossless_jpeg_stream(width as u16, height as u16, 1, &samples);
    let dng = build_compressed_dng(width, height, 7, &jpeg, CfaPattern::Rggb);

    assert!(DngReader::is_dng(&dng));
    let image = DngReader::read(&dng).expect("decode lossless-JPEG DNG");
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.raw_data.len(), (width * height) as usize);
    // Lossless: the raw data must match the originals exactly.
    assert_eq!(image.raw_data, samples);
}

#[test]
fn test_dng_lossless_jpeg_two_component_cfa() {
    // DNG-style 2-component packing: a 12x4 CFA is stored as a 6x4x2 JPEG.
    let cfa_width = 12u32;
    let cfa_height = 4u32;
    let jpeg_w = (cfa_width / 2) as u16;
    let components = 2u8;
    // Component-interleaved samples [c0,c1, c0,c1, ...].
    let jpeg_samples: Vec<u16> = (0..(jpeg_w as u32 * cfa_height * 2))
        .map(|i| (i as u16).wrapping_mul(91).wrapping_add(2000))
        .collect();
    let jpeg = build_lossless_jpeg_stream(jpeg_w, cfa_height as u16, components, &jpeg_samples);
    let dng = build_compressed_dng(cfa_width, cfa_height, 7, &jpeg, CfaPattern::Rggb);

    let image = DngReader::read(&dng).expect("decode 2-component lossless-JPEG DNG");
    assert_eq!(image.width, cfa_width);
    assert_eq!(image.height, cfa_height);
    assert_eq!(image.raw_data.len(), (cfa_width * cfa_height) as usize);

    // CFA column cx is component cx%2 of JPEG column cx/2.
    for y in 0..cfa_height as usize {
        for cx in 0..cfa_width as usize {
            let comp = cx % 2;
            let jx = cx / 2;
            let expected = jpeg_samples[(y * jpeg_w as usize + jx) * 2 + comp];
            assert_eq!(
                image.raw_data[y * cfa_width as usize + cx],
                expected,
                "CFA mismatch at ({cx},{y})"
            );
        }
    }
}

#[test]
fn test_dng_lossless_jpeg_roundtrip_via_tempfile() {
    // Exercise the file-backed path using std::env::temp_dir().
    let width = 6u32;
    let height = 6u32;
    let samples: Vec<u16> = (0..(width * height))
        .map(|i| 8000u16.wrapping_add((i as u16).wrapping_mul(257)))
        .collect();
    let jpeg = build_lossless_jpeg_stream(width as u16, height as u16, 1, &samples);
    let dng = build_compressed_dng(width, height, 7, &jpeg, CfaPattern::Grbg);

    let mut path = std::env::temp_dir();
    path.push(format!("oximedia_dng_lossless_{}.dng", std::process::id()));
    std::fs::write(&path, &dng).expect("write temp DNG");
    let read_bytes = std::fs::read(&path).expect("read temp DNG");
    let _ = std::fs::remove_file(&path);

    let image = DngReader::read(&read_bytes).expect("decode lossless-JPEG DNG from file");
    assert_eq!(image.raw_data, samples);
    assert_eq!(image.metadata.cfa_pattern, CfaPattern::Grbg);
}

#[test]
fn test_dng_lossy_dng_baseline_jpeg() {
    // Lossy DNG: raw mosaic stored as a baseline 8-bit DCT JPEG.
    let width = 16u32;
    let height = 16u32;
    // A smooth gamma-encoded-style ramp survives DCT compression well.
    let pixels: Vec<u8> = (0..(width * height))
        .map(|i| {
            let x = i % width;
            let y = i / width;
            ((x * 8 + y * 6) % 230 + 12) as u8
        })
        .collect();

    let frame = ImageFrame::new(
        0,
        width,
        height,
        PixelType::U8,
        1,
        ColorSpace::Luma,
        ImageData::interleaved(pixels.clone()),
    );
    let encoder = crate::jpeg::JpegEncoder::new(crate::jpeg::JpegQuality::high());
    let jpeg = encoder.encode(&frame).expect("encode baseline JPEG");
    let dng = build_compressed_dng(width, height, 34892, &jpeg, CfaPattern::Rggb);

    assert!(DngReader::is_dng(&dng));
    let image = DngReader::read(&dng).expect("decode lossy DNG");
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.raw_data.len(), (width * height) as usize);

    // Lossy: values are widened from 8-bit; allow DCT round-trip error.
    let mut max_err = 0i32;
    for (decoded, original) in image.raw_data.iter().zip(pixels.iter()) {
        max_err = max_err.max((i32::from(*decoded) - i32::from(*original)).abs());
        // The decoded samples are 8-bit-derived, so they fit in a byte.
        assert!(*decoded <= 255, "lossy DNG sample exceeds 8-bit range");
    }
    assert!(
        max_err < 45,
        "lossy DNG round-trip error too large: {max_err}"
    );
}

#[test]
fn test_dng_lossy_dng_via_tempfile() {
    let width = 16u32;
    let height = 8u32;
    let pixels: Vec<u8> = (0..(width * height))
        .map(|i| ((i * 5) % 200 + 20) as u8)
        .collect();
    let frame = ImageFrame::new(
        0,
        width,
        height,
        PixelType::U8,
        1,
        ColorSpace::Luma,
        ImageData::interleaved(pixels),
    );
    let encoder = crate::jpeg::JpegEncoder::new(crate::jpeg::JpegQuality::high());
    let jpeg = encoder.encode(&frame).expect("encode");
    let dng = build_compressed_dng(width, height, 34892, &jpeg, CfaPattern::Bggr);

    let mut path = std::env::temp_dir();
    path.push(format!("oximedia_dng_lossy_{}.dng", std::process::id()));
    std::fs::write(&path, &dng).expect("write temp DNG");
    let read_bytes = std::fs::read(&path).expect("read temp DNG");
    let _ = std::fs::remove_file(&path);

    let image = DngReader::read(&read_bytes).expect("decode lossy DNG from file");
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
}

#[test]
fn test_dng_lossless_jpeg_rejects_corrupt_strip() {
    // A strip that is not a JPEG at all must surface a decode error.
    let dng = build_compressed_dng(4, 4, 7, &[0x00, 0x11, 0x22, 0x33], CfaPattern::Rggb);
    assert!(DngReader::read(&dng).is_err());
}
