// Tests for the OpenEXR 2.0 multi-part format implementation.

use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{Cursor, Seek, SeekFrom, Write};

use crate::error::ImageError;
use super::{EXR_MAGIC, EXR_VERSION};
use super::types::{
    ExrBox2i, ExrChannel, ExrChannelType, ExrCompression, ExrPart, ExrPartType,
};
use super::write::write_nul_string;
use super::MultiPartExr;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn rgba_channels() -> Vec<ExrChannel> {
    vec![
        ExrChannel::float("R"),
        ExrChannel::float("G"),
        ExrChannel::float("B"),
        ExrChannel::float("A"),
    ]
}

fn depth_channels() -> Vec<ExrChannel> {
    vec![ExrChannel::float("Z")]
}

fn make_rgba_part(name: &str, w: u32, h: u32) -> ExrPart {
    let window = ExrBox2i::from_dims(w, h);
    let mut part = ExrPart::new(
        name,
        ExrPartType::ScanlineImage,
        rgba_channels(),
        window,
        window,
        ExrCompression::None,
    );
    // Fill with a simple gradient so round-trip values are non-trivial
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize * 4;
            part.pixels[idx] = x as f32 / w as f32;
            part.pixels[idx + 1] = y as f32 / h as f32;
            part.pixels[idx + 2] = 0.5;
            part.pixels[idx + 3] = 1.0;
        }
    }
    part
}

fn make_depth_part(name: &str, w: u32, h: u32, fill: f32) -> ExrPart {
    let window = ExrBox2i::from_dims(w, h);
    let mut part = ExrPart::new(
        name,
        ExrPartType::ScanlineImage,
        depth_channels(),
        window,
        window,
        ExrCompression::None,
    );
    part.pixels.fill(fill);
    part
}

// ── ExrBox2i ──────────────────────────────────────────────────────────────────

#[test]
fn test_box2i_dimensions() {
    let b = ExrBox2i {
        x_min: 0,
        y_min: 0,
        x_max: 99,
        y_max: 49,
    };
    assert_eq!(b.width(), 100);
    assert_eq!(b.height(), 50);
    assert_eq!(b.pixel_count(), 5000);
}

#[test]
fn test_box2i_from_dims() {
    let b = ExrBox2i::from_dims(4, 4);
    assert_eq!(b.x_min, 0);
    assert_eq!(b.x_max, 3);
    assert_eq!(b.width(), 4);
}

#[test]
fn test_box2i_zero_dims() {
    let b = ExrBox2i::from_dims(0, 0);
    assert_eq!(b.width(), 0);
    assert_eq!(b.height(), 0);
}

// ── Channel type ──────────────────────────────────────────────────────────────

#[test]
fn test_channel_type_wire_round_trip() {
    for code in [0u32, 1, 2] {
        let ct = ExrChannelType::from_wire_code(code).expect("valid code");
        assert_eq!(ct.wire_code(), code);
    }
}

#[test]
fn test_channel_type_bytes_per_sample() {
    assert_eq!(ExrChannelType::Half.bytes_per_sample(), 2);
    assert_eq!(ExrChannelType::Float.bytes_per_sample(), 4);
    assert_eq!(ExrChannelType::Uint.bytes_per_sample(), 4);
}

#[test]
fn test_channel_type_unknown_code() {
    assert!(ExrChannelType::from_wire_code(99).is_err());
}

// ── Compression ───────────────────────────────────────────────────────────────

#[test]
fn test_compression_wire_round_trip() {
    for code in 0..=9u8 {
        let c = ExrCompression::from_wire_code(code).expect("valid");
        assert_eq!(c.wire_code(), code);
    }
}

#[test]
fn test_compression_unknown_code() {
    assert!(ExrCompression::from_wire_code(42).is_err());
}

#[test]
fn test_compression_scanlines_per_block() {
    assert_eq!(ExrCompression::None.scanlines_per_block(), 1);
    assert_eq!(ExrCompression::Zip.scanlines_per_block(), 16);
    assert_eq!(ExrCompression::Dwab.scanlines_per_block(), 256);
}

// ── ExrPartType ───────────────────────────────────────────────────────────────

#[test]
fn test_part_type_round_trip() {
    for (s, expected) in [
        ("scanlineimage", ExrPartType::ScanlineImage),
        ("tiledimage", ExrPartType::TiledImage),
        ("deepscanline", ExrPartType::DeepScanline),
        ("deeptile", ExrPartType::DeepTile),
    ] {
        let parsed = ExrPartType::from_str(s).expect("valid");
        assert_eq!(parsed, expected);
        assert_eq!(parsed.as_str(), s);
    }
}

#[test]
fn test_part_type_unknown() {
    assert!(ExrPartType::from_str("unknowntype").is_err());
}

// ── ExrPart validation ────────────────────────────────────────────────────────

#[test]
fn test_part_validate_ok() {
    let part = make_rgba_part("beauty", 8, 8);
    part.validate().expect("should be valid");
}

#[test]
fn test_part_validate_wrong_pixel_count() {
    let window = ExrBox2i::from_dims(4, 4);
    let part = ExrPart {
        name: "bad".to_string(),
        part_type: ExrPartType::ScanlineImage,
        channels: rgba_channels(),
        data_window: window,
        display_window: window,
        compression: ExrCompression::None,
        pixels: vec![0.0; 10], // wrong length
        width: 4,
        height: 4,
    };
    assert!(part.validate().is_err());
}

#[test]
fn test_part_sample_access() {
    let mut part = make_rgba_part("beauty", 4, 4);
    part.set_sample(2, 3, 0, 0.99).expect("set");
    let v = part.get_sample(2, 3, 0).expect("get");
    assert!((v - 0.99).abs() < 1e-6);
}

#[test]
fn test_part_sample_out_of_bounds_x() {
    let part = make_rgba_part("a", 4, 4);
    assert!(part.get_sample(4, 0, 0).is_err());
}

#[test]
fn test_part_sample_out_of_bounds_y() {
    let part = make_rgba_part("a", 4, 4);
    assert!(part.get_sample(0, 4, 0).is_err());
}

#[test]
fn test_part_sample_bad_channel_index() {
    let part = make_rgba_part("a", 4, 4);
    assert!(part.get_sample(0, 0, 99).is_err());
}

// ── MultiPartExr document operations ─────────────────────────────────────────

#[test]
fn test_part_by_name_found() {
    let mut doc = MultiPartExr::new();
    doc.add_part(make_rgba_part("color", 4, 4));
    doc.add_part(make_depth_part("depth", 4, 4, 1000.0));
    assert!(doc.part_by_name("color").is_some());
    assert!(doc.part_by_name("depth").is_some());
    assert!(doc.part_by_name("nonexistent").is_none());
}

#[test]
fn test_part_by_name_mut() {
    let mut doc = MultiPartExr::new();
    doc.add_part(make_rgba_part("color", 4, 4));
    let part = doc.part_by_name_mut("color").expect("exists");
    part.pixels[0] = 42.0;
    assert!((doc.parts[0].pixels[0] - 42.0).abs() < 1e-6);
}

#[test]
fn test_validate_duplicate_names() {
    let mut doc = MultiPartExr::new();
    doc.add_part(make_rgba_part("color", 4, 4));
    doc.add_part(make_rgba_part("color", 4, 4));
    assert!(doc.validate().is_err());
}

#[test]
fn test_validate_empty_document() {
    let doc = MultiPartExr::new();
    assert!(doc.validate().is_err());
}

// ── Scanline encoding helpers ─────────────────────────────────────────────────

#[test]
fn test_encode_decode_scanline_float() {
    use super::write::encode_scanline;
    use super::parse::decode_scanline;
    let channels = vec![
        ExrChannel::float("R"),
        ExrChannel::float("G"),
        ExrChannel::float("B"),
    ];
    let pixels: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect(); // 4 px × 3 ch
    let encoded = encode_scanline(&pixels, 4, &channels);
    let decoded = decode_scanline(&encoded, 4, &channels).expect("decode");
    assert_eq!(decoded.len(), pixels.len());
    for (a, b) in pixels.iter().zip(decoded.iter()) {
        assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
    }
}

#[test]
fn test_encode_decode_scanline_half() {
    use super::write::encode_scanline;
    use super::parse::decode_scanline;
    let channels = vec![ExrChannel::half("Y")];
    let pixels: Vec<f32> = vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 10.0];
    let encoded = encode_scanline(&pixels, 8, &channels);
    let decoded = decode_scanline(&encoded, 8, &channels).expect("decode");
    assert_eq!(decoded.len(), pixels.len());
    // Half has ~3 decimal digits of precision
    for (a, b) in pixels.iter().zip(decoded.iter()) {
        assert!((a - b).abs() < 0.01, "half mismatch: {a} vs {b}");
    }
}

#[test]
fn test_encode_decode_scanline_uint() {
    use super::write::encode_scanline;
    use super::parse::decode_scanline;
    let channels = vec![ExrChannel {
        name: "ID".to_string(),
        channel_type: ExrChannelType::Uint,
        x_sampling: 1,
        y_sampling: 1,
        linear: true,
    }];
    let pixels: Vec<f32> = vec![0.0, 1.0, 255.0, 65535.0];
    let encoded = encode_scanline(&pixels, 4, &channels);
    let decoded = decode_scanline(&encoded, 4, &channels).expect("decode");
    for (a, b) in pixels.iter().zip(decoded.iter()) {
        assert!((a - b).abs() < 1.0, "uint mismatch: {a} vs {b}");
    }
}

// ── Full round-trip ───────────────────────────────────────────────────────────

/// Round-trip a 2-part EXR (RGBA + depth) and verify channel names and
/// pixel values survive serialisation.
#[test]
fn test_multipart_roundtrip() {
    let w = 8u32;
    let h = 6u32;
    let rgba = make_rgba_part("color", w, h);
    let depth = make_depth_part("depth", w, h, 5.0);

    // Remember original pixel values
    let orig_rgba = rgba.pixels.clone();
    let orig_depth = depth.pixels.clone();

    let mut doc = MultiPartExr::new();
    doc.add_part(rgba);
    doc.add_part(depth);

    let bytes = doc.to_bytes().expect("to_bytes");
    assert!(!bytes.is_empty());

    // Verify multi-part flag
    let is_mp = MultiPartExr::is_multipart_bytes(&bytes).expect("check flag");
    assert!(is_mp, "serialised bytes should have multi-part flag");

    let roundtrip = MultiPartExr::from_bytes(&bytes).expect("from_bytes");
    assert_eq!(roundtrip.parts.len(), 2);

    // Part 0: color (RGBA)
    let rt_rgba = &roundtrip.parts[0];
    assert_eq!(rt_rgba.name, "color");
    assert_eq!(rt_rgba.channels.len(), 4);
    assert_eq!(
        rt_rgba
            .channels
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["R", "G", "B", "A"]
    );
    assert_eq!(rt_rgba.width, w);
    assert_eq!(rt_rgba.height, h);
    assert_eq!(rt_rgba.pixels.len(), orig_rgba.len());
    for (a, b) in orig_rgba.iter().zip(rt_rgba.pixels.iter()) {
        assert!((a - b).abs() < 1e-5, "RGBA pixel mismatch: {a} vs {b}");
    }

    // Part 1: depth (Z)
    let rt_depth = &roundtrip.parts[1];
    assert_eq!(rt_depth.name, "depth");
    assert_eq!(rt_depth.channels.len(), 1);
    assert_eq!(rt_depth.channels[0].name, "Z");
    assert_eq!(rt_depth.width, w);
    assert_eq!(rt_depth.height, h);
    for v in &rt_depth.pixels {
        assert!((v - 5.0).abs() < 1e-5, "depth pixel mismatch: {v}");
    }
    assert_eq!(rt_depth.pixels.len(), orig_depth.len());
}

/// A 3-part document round-trips and `part_by_name` locates the middle part.
#[test]
fn test_part_by_name_three_parts() {
    let window = ExrBox2i::from_dims(2, 2);
    let make = |name: &str| {
        ExrPart::new(
            name,
            ExrPartType::ScanlineImage,
            vec![ExrChannel::float("V")],
            window,
            window,
            ExrCompression::None,
        )
    };
    let mut doc = MultiPartExr::new();
    doc.add_part(make("diffuse"));
    doc.add_part(make("specular"));
    doc.add_part(make("ao"));

    let bytes = doc.to_bytes().expect("to_bytes");
    let rt = MultiPartExr::from_bytes(&bytes).expect("from_bytes");

    assert!(rt.part_by_name("diffuse").is_some());
    assert!(rt.part_by_name("specular").is_some());
    assert!(rt.part_by_name("ao").is_some());
    assert!(rt.part_by_name("missing").is_none());

    let sp = rt.part_by_name("specular").expect("found");
    assert_eq!(sp.name, "specular");
    assert_eq!(sp.channel_count(), 1);
}

/// Parsing a non-multi-part EXR (single-part, no 0x1000 flag) via
/// `from_bytes` returns it as a single-part `MultiPartExr`.
#[test]
fn test_single_part_detected_as_single() {
    // Build a minimal single-part EXR in memory
    let mut bytes: Vec<u8> = Vec::new();
    let mut cur = Cursor::new(&mut bytes);

    // Magic + version (no multi-part flag)
    cur.write_u32::<LittleEndian>(EXR_MAGIC).unwrap();
    let version_word: u32 = EXR_VERSION as u32; // no flags
    cur.write_u32::<LittleEndian>(version_word).unwrap();

    // Single channel: R (float)
    write_nul_string(&mut cur, "channels").unwrap();
    write_nul_string(&mut cur, "chlist").unwrap();
    // chlist: "R\0 type(4) pLinear(1) pad(3) xSamp(4) ySamp(4) \0" = 4+1+4+1+3+4+4+1 = 22
    let ch_data: &[u8] =
        b"R\x00\x02\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00";
    cur.write_u32::<LittleEndian>(ch_data.len() as u32).unwrap();
    cur.write_all(ch_data).unwrap();

    // compression: none
    write_nul_string(&mut cur, "compression").unwrap();
    write_nul_string(&mut cur, "compression").unwrap();
    cur.write_u32::<LittleEndian>(1).unwrap();
    cur.write_u8(0).unwrap(); // None

    // dataWindow: (0,0,1,1)
    write_nul_string(&mut cur, "dataWindow").unwrap();
    write_nul_string(&mut cur, "box2i").unwrap();
    cur.write_u32::<LittleEndian>(16).unwrap();
    for v in [0i32, 0, 1, 1] {
        cur.write_i32::<LittleEndian>(v).unwrap();
    }

    // displayWindow: same
    write_nul_string(&mut cur, "displayWindow").unwrap();
    write_nul_string(&mut cur, "box2i").unwrap();
    cur.write_u32::<LittleEndian>(16).unwrap();
    for v in [0i32, 0, 1, 1] {
        cur.write_i32::<LittleEndian>(v).unwrap();
    }

    // lineOrder
    write_nul_string(&mut cur, "lineOrder").unwrap();
    write_nul_string(&mut cur, "lineOrder").unwrap();
    cur.write_u32::<LittleEndian>(1).unwrap();
    cur.write_u8(0).unwrap();

    // pixelAspectRatio
    write_nul_string(&mut cur, "pixelAspectRatio").unwrap();
    write_nul_string(&mut cur, "float").unwrap();
    cur.write_u32::<LittleEndian>(4).unwrap();
    cur.write_f32::<LittleEndian>(1.0).unwrap();

    // screenWindowCenter
    write_nul_string(&mut cur, "screenWindowCenter").unwrap();
    write_nul_string(&mut cur, "v2f").unwrap();
    cur.write_u32::<LittleEndian>(8).unwrap();
    cur.write_f32::<LittleEndian>(0.0).unwrap();
    cur.write_f32::<LittleEndian>(0.0).unwrap();

    // screenWindowWidth
    write_nul_string(&mut cur, "screenWindowWidth").unwrap();
    write_nul_string(&mut cur, "float").unwrap();
    cur.write_u32::<LittleEndian>(4).unwrap();
    cur.write_f32::<LittleEndian>(1.0).unwrap();

    // End of header
    cur.write_u8(0).unwrap();

    // Offset table: 2 scanlines (height=2)
    let header_end = cur.position();
    // scanline offsets placeholder — fill after
    let offset_table_pos = cur.position();
    cur.write_u64::<LittleEndian>(0).unwrap(); // y=0
    cur.write_u64::<LittleEndian>(0).unwrap(); // y=1

    // Scanline 0: y=0, 2 pixels × 1 ch × 4 bytes = 8 bytes
    let sl0_pos = cur.position();
    cur.write_i32::<LittleEndian>(0).unwrap(); // y coord
    cur.write_u32::<LittleEndian>(8).unwrap(); // data size
    cur.write_all(&[0x00, 0x00, 0x80, 0x3f]).unwrap(); // f32 1.0
    cur.write_all(&[0x00, 0x00, 0x00, 0x40]).unwrap(); // f32 2.0

    // Scanline 1: y=1
    let sl1_pos = cur.position();
    cur.write_i32::<LittleEndian>(1).unwrap();
    cur.write_u32::<LittleEndian>(8).unwrap();
    cur.write_all(&[0x00, 0x00, 0x40, 0x40]).unwrap(); // f32 3.0
    cur.write_all(&[0x00, 0x00, 0x80, 0x40]).unwrap(); // f32 4.0

    // Back-fill offsets
    cur.seek(SeekFrom::Start(offset_table_pos)).unwrap();
    cur.write_u64::<LittleEndian>(sl0_pos).unwrap();
    cur.write_u64::<LittleEndian>(sl1_pos).unwrap();

    let _ = (cur, header_end);

    // Parse it
    let doc = MultiPartExr::from_bytes(&bytes).expect("parse single-part");
    assert_eq!(doc.parts.len(), 1, "single-part should yield 1 part");

    let part = &doc.parts[0];
    assert_eq!(part.channels.len(), 1);
    assert_eq!(part.channels[0].name, "R");
    assert_eq!(part.width, 2);
    assert_eq!(part.height, 2);
    assert_eq!(part.pixels.len(), 4);
    // Verify pixel values
    assert!((part.pixels[0] - 1.0).abs() < 1e-5);
    assert!((part.pixels[1] - 2.0).abs() < 1e-5);
    assert!((part.pixels[2] - 3.0).abs() < 1e-5);
    assert!((part.pixels[3] - 4.0).abs() < 1e-5);
}

/// is_exr and is_multipart_bytes work correctly.
#[test]
fn test_is_exr_helpers() {
    let non_exr = b"PNG\r\n\x1a\n";
    assert!(!MultiPartExr::is_exr(non_exr));

    let mut doc = MultiPartExr::new();
    doc.add_part(make_rgba_part("p", 2, 2));
    let bytes = doc.to_bytes().expect("to_bytes");

    assert!(MultiPartExr::is_exr(&bytes));
    let is_mp = MultiPartExr::is_multipart_bytes(&bytes).expect("ok");
    assert!(is_mp);

    assert!(MultiPartExr::is_multipart_bytes(b"short").is_err());
}

/// Writing a Tiled, DeepScanline, or DeepTile part returns Unsupported.
#[test]
fn test_write_unsupported_part_type() {
    let window = ExrBox2i::from_dims(4, 4);
    for pt in [
        ExrPartType::TiledImage,
        ExrPartType::DeepScanline,
        ExrPartType::DeepTile,
    ] {
        let part = ExrPart {
            name: "p".to_string(),
            part_type: pt,
            channels: vec![ExrChannel::float("R")],
            data_window: window,
            display_window: window,
            compression: ExrCompression::None,
            pixels: vec![0.0; 16],
            width: 4,
            height: 4,
        };
        let doc = MultiPartExr { parts: vec![part] };
        let result = doc.to_bytes();
        assert!(result.is_err(), "expected Err for non-scanline part type");
        match result {
            Err(ImageError::Unsupported(_)) => {}
            Err(e) => panic!("expected Unsupported, got {e:?}"),
            Ok(_) => panic!("expected error"),
        }
    }
}

/// RLE-compressed multi-part round-trip.
#[test]
fn test_rle_compressed_roundtrip() {
    let window = ExrBox2i::from_dims(4, 4);
    let mut part = ExrPart::new(
        "rle_test",
        ExrPartType::ScanlineImage,
        vec![ExrChannel::float("R")],
        window,
        window,
        ExrCompression::Rle,
    );
    // Fill with a constant value (RLE compresses this very well)
    part.pixels.fill(0.5);

    let mut doc = MultiPartExr::new();
    doc.add_part(part);
    let bytes = doc.to_bytes().expect("to_bytes rle");
    let rt = MultiPartExr::from_bytes(&bytes).expect("from_bytes rle");
    for v in &rt.parts[0].pixels {
        assert!((v - 0.5).abs() < 1e-5);
    }
}

/// Zip-compressed multi-part round-trip.
#[test]
fn test_zip_compressed_roundtrip() {
    let window = ExrBox2i::from_dims(8, 8);
    let mut part = ExrPart::new(
        "zip_test",
        ExrPartType::ScanlineImage,
        vec![ExrChannel::float("R"), ExrChannel::float("G")],
        window,
        window,
        ExrCompression::ZipSingle,
    );
    for (i, v) in part.pixels.iter_mut().enumerate() {
        *v = (i as f32) / 128.0;
    }
    let orig = part.pixels.clone();
    let mut doc = MultiPartExr::new();
    doc.add_part(part);
    let bytes = doc.to_bytes().expect("to_bytes zip");
    let rt = MultiPartExr::from_bytes(&bytes).expect("from_bytes zip");
    assert_eq!(rt.parts[0].pixels.len(), orig.len());
    for (a, b) in orig.iter().zip(rt.parts[0].pixels.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

/// Half-float channels survive round-trip with acceptable precision.
#[test]
fn test_half_channel_roundtrip() {
    let window = ExrBox2i::from_dims(4, 4);
    let mut part = ExrPart::new(
        "half_test",
        ExrPartType::ScanlineImage,
        vec![ExrChannel::half("H")],
        window,
        window,
        ExrCompression::None,
    );
    for (i, v) in part.pixels.iter_mut().enumerate() {
        *v = i as f32 * 0.25;
    }
    let orig = part.pixels.clone();
    let mut doc = MultiPartExr::new();
    doc.add_part(part);
    let bytes = doc.to_bytes().expect("to_bytes half");
    let rt = MultiPartExr::from_bytes(&bytes).expect("from_bytes half");
    for (a, b) in orig.iter().zip(rt.parts[0].pixels.iter()) {
        // Half precision: ~0.001 relative error
        assert!((a - b).abs() < 0.01, "half mismatch: {a} vs {b}");
    }
}

/// A 1×1 image survives round-trip.
#[test]
fn test_minimal_1x1_roundtrip() {
    let window = ExrBox2i::from_dims(1, 1);
    let mut part = ExrPart::new(
        "tiny",
        ExrPartType::ScanlineImage,
        vec![ExrChannel::float("R")],
        window,
        window,
        ExrCompression::None,
    );
    part.pixels[0] = std::f32::consts::PI;
    let mut doc = MultiPartExr::new();
    doc.add_part(part);
    let bytes = doc.to_bytes().expect("to_bytes 1x1");
    let rt = MultiPartExr::from_bytes(&bytes).expect("from_bytes 1x1");
    assert!((rt.parts[0].pixels[0] - std::f32::consts::PI).abs() < 1e-6);
}

/// Document default is equivalent to new().
#[test]
fn test_document_default() {
    let doc: MultiPartExr = MultiPartExr::default();
    assert_eq!(doc.part_count(), 0);
}
