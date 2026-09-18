//! Tests for the TIFF implementation.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use super::*;
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::Write;

/// Builds a minimal tiled TIFF binary (little-endian, RGB 8-bit, no compression).
///
/// Image:  4 × 4 pixels
/// Tiles:  2 × 2 pixels → 2 tiles across × 2 tiles down = 4 tiles
///
/// Tile fill values (R,G,B for every pixel in that tile):
///   Tile 0 (top-left):     0x01, 0x02, 0x03
///   Tile 1 (top-right):    0x04, 0x05, 0x06
///   Tile 2 (bottom-left):  0x07, 0x08, 0x09
///   Tile 3 (bottom-right): 0x0A, 0x0B, 0x0C
fn build_tiled_tiff() -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();

    // ---- TIFF header ----
    buf.write_u16::<LittleEndian>(TIFF_MAGIC_LE).expect("magic");
    buf.write_u16::<LittleEndian>(TIFF_VERSION)
        .expect("version");
    // IFD offset placeholder at bytes 4..8 — patched below.
    buf.write_u32::<LittleEndian>(0)
        .expect("ifd offset placeholder");

    // ---- Raw tile pixel data (no compression) ----
    // 4 tiles × 2×2 pixels × 3 bytes/pixel = 48 bytes
    // Each tile is 12 bytes (4 pixels × 3 channels)
    let tile_data_base: u32 = buf.len() as u32; // = 8

    let tiles: &[(u8, u8, u8)] = &[
        (0x01, 0x02, 0x03),
        (0x04, 0x05, 0x06),
        (0x07, 0x08, 0x09),
        (0x0A, 0x0B, 0x0C),
    ];

    // Per-tile byte offsets within the file
    let tile_offsets: Vec<u32> = (0..4u32).map(|i| tile_data_base + i * 12).collect();

    for &(r, g, b) in tiles {
        for _ in 0..4u8 {
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
    }

    // ---- Out-of-line arrays ----

    // BitsPerSample: [8, 8, 8] stored out-of-line (3 × u16 = 6 bytes > 4)
    let bits_per_sample_pos: u32 = buf.len() as u32;
    for _ in 0..3u8 {
        buf.write_u16::<LittleEndian>(8)
            .expect("bits_per_sample entry");
    }

    // TileOffsets: 4 × u32
    let tile_offsets_arr_pos: u32 = buf.len() as u32;
    for &off in &tile_offsets {
        buf.write_u32::<LittleEndian>(off)
            .expect("tile offset entry");
    }

    // TileByteCounts: 4 × u32 (all = 12)
    let tile_byte_counts_arr_pos: u32 = buf.len() as u32;
    for _ in 0..4u32 {
        buf.write_u32::<LittleEndian>(12)
            .expect("tile byte count entry");
    }

    // ---- IFD ----
    let ifd_pos: u32 = buf.len() as u32;
    // 10 tags
    buf.write_u16::<LittleEndian>(10).expect("tag count");

    // Inline macro to write one 12-byte IFD entry (LE).
    macro_rules! tag {
        ($tag:expr, $dtype:expr, $count:expr, $val:expr) => {{
            buf.write_u16::<LittleEndian>($tag).expect("ifd tag");
            buf.write_u16::<LittleEndian>($dtype).expect("ifd dtype");
            buf.write_u32::<LittleEndian>($count).expect("ifd count");
            buf.write_u32::<LittleEndian>($val).expect("ifd value");
        }};
    }

    tag!(256, 4, 1, 4); // ImageWidth = 4
    tag!(257, 4, 1, 4); // ImageLength = 4
                        // BitsPerSample: count=3 (one per channel), offset→bits_per_sample_pos
                        // total size = 3×2=6 > 4 → stored out-of-line
    tag!(258, 3, 3, bits_per_sample_pos); // BitsPerSample = [8,8,8]
    tag!(259, 3, 1, 1); // Compression = 1 (None)
    tag!(262, 3, 1, 2); // PhotometricInterpretation = 2 (RGB)
    tag!(277, 3, 1, 3); // SamplesPerPixel = 3
    tag!(322, 3, 1, 2); // TileWidth = 2
    tag!(323, 3, 1, 2); // TileLength = 2
    tag!(324, 4, 4, tile_offsets_arr_pos); // TileOffsets
    tag!(325, 4, 4, tile_byte_counts_arr_pos); // TileByteCounts

    // Next-IFD pointer = 0 (end)
    buf.write_u32::<LittleEndian>(0).expect("next ifd");

    // Patch IFD offset at header bytes [4..8]
    buf[4..8].copy_from_slice(&ifd_pos.to_le_bytes());

    buf
}

#[test]
fn test_read_tiled_tiff_2x2_tiles() {
    let tiff_bytes = build_tiled_tiff();

    let mut temp_path = std::env::temp_dir();
    temp_path.push("oximedia_test_tiled_tiff_2x2.tif");

    {
        let mut f = std::fs::File::create(&temp_path).expect("should create temp file");
        f.write_all(&tiff_bytes).expect("should write TIFF bytes");
    }

    let frame = read_tiff(&temp_path, 0).expect("should read tiled TIFF");

    assert_eq!(frame.width, 4, "width mismatch");
    assert_eq!(frame.height, 4, "height mismatch");

    let data = frame.data.as_slice().expect("data should be interleaved");
    // 4 × 4 × 3 bytes = 48
    assert_eq!(data.len(), 48, "data length mismatch");

    // Stride = 4 cols × 3 bytes = 12 bytes/row
    // Tile 0 (top-left):  pixel (col=0, row=0) → byte offset = 0
    assert_eq!(data[0], 0x01, "R at (0,0)");
    assert_eq!(data[1], 0x02, "G at (0,0)");
    assert_eq!(data[2], 0x03, "B at (0,0)");

    // Tile 1 (top-right): pixel (col=2, row=0) → byte offset = 2*3 = 6
    assert_eq!(data[6], 0x04, "R at (2,0)");
    assert_eq!(data[7], 0x05, "G at (2,0)");
    assert_eq!(data[8], 0x06, "B at (2,0)");

    // Tile 2 (bottom-left): pixel (col=0, row=2) → byte offset = 2*12 = 24
    assert_eq!(data[24], 0x07, "R at (0,2)");
    assert_eq!(data[25], 0x08, "G at (0,2)");
    assert_eq!(data[26], 0x09, "B at (0,2)");

    // Tile 3 (bottom-right): pixel (col=2, row=2) → byte offset = 2*12 + 2*3 = 30
    assert_eq!(data[30], 0x0A, "R at (2,2)");
    assert_eq!(data[31], 0x0B, "G at (2,2)");
    assert_eq!(data[32], 0x0C, "B at (2,2)");

    let _ = std::fs::remove_file(&temp_path);
}

// -----------------------------------------------------------------------
// JPEG-in-TIFF round trip (write_tiff → read_tiff)
// -----------------------------------------------------------------------

#[test]
fn test_jpeg_tiff_roundtrip_rgb() {
    // A smooth 32x32 RGB gradient survives baseline JPEG compression with
    // a bounded per-channel error.
    let width = 32u32;
    let height = 32u32;
    let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push((x * 8) as u8);
            pixels.push((y * 8) as u8);
            pixels.push(((x + y) * 4) as u8);
        }
    }
    let frame = ImageFrame::new(
        0,
        width,
        height,
        PixelType::U8,
        3,
        ColorSpace::Srgb,
        ImageData::interleaved(pixels.clone()),
    );

    let mut path = std::env::temp_dir();
    path.push("oximedia_tiff_jpeg_rgb_roundtrip.tif");
    write_tiff(&path, &frame, TiffCompression::Jpeg).expect("write JPEG TIFF");

    let decoded = read_tiff(&path, 0).expect("read JPEG TIFF");
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    let data = decoded.data.as_slice().expect("interleaved");
    assert_eq!(data.len(), pixels.len());

    // Baseline JPEG is lossy; check the mean absolute error stays small.
    let total: u64 = data
        .iter()
        .zip(&pixels)
        .map(|(&a, &b)| u64::from(a.abs_diff(b)))
        .sum();
    let mae = total as f64 / data.len() as f64;
    assert!(mae < 24.0, "JPEG mean abs error too high: {mae}");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_jpeg_tiff_roundtrip_grey() {
    // A flat grey image must round-trip near-exactly.
    let width = 24u32;
    let height = 16u32;
    let pixels = vec![137u8; (width * height) as usize];
    let frame = ImageFrame::new(
        0,
        width,
        height,
        PixelType::U8,
        1,
        ColorSpace::Luma,
        ImageData::interleaved(pixels.clone()),
    );

    let mut path = std::env::temp_dir();
    path.push("oximedia_tiff_jpeg_grey_roundtrip.tif");
    write_tiff(&path, &frame, TiffCompression::Jpeg).expect("write grey JPEG TIFF");

    let decoded = read_tiff(&path, 0).expect("read grey JPEG TIFF");
    let data = decoded.data.as_slice().expect("interleaved");
    assert_eq!(data.len(), pixels.len());
    for &v in data {
        assert!(v.abs_diff(137) <= 4, "flat grey JPEG drifted too far: {v}");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_jpeg_tiff_emits_jpegtables() {
    // A JPEG-compressed TIFF must carry the JPEGTables (347) tag and the
    // tag must be re-readable.
    let width = 8u32;
    let height = 8u32;
    let pixels = vec![64u8; (width * height * 3) as usize];
    let frame = ImageFrame::new(
        0,
        width,
        height,
        PixelType::U8,
        3,
        ColorSpace::Srgb,
        ImageData::interleaved(pixels),
    );
    let mut path = std::env::temp_dir();
    path.push("oximedia_tiff_jpeg_tables.tif");
    write_tiff(&path, &frame, TiffCompression::Jpeg).expect("write");

    // Re-parse the IFD directly and confirm JPEGTables is present.
    let mut file = File::open(&path).expect("open");
    let _magic = file.read_u16::<BigEndian>().expect("magic");
    let endian = Endian::native();
    let _ver = match endian {
        Endian::Big => file.read_u16::<BigEndian>().expect("ver"),
        Endian::Little => file.read_u16::<LittleEndian>().expect("ver"),
    };
    let ifd_off = match endian {
        Endian::Big => u64::from(file.read_u32::<BigEndian>().expect("ifd")),
        Endian::Little => u64::from(file.read_u32::<LittleEndian>().expect("ifd")),
    };
    let info = read_ifd(&mut file, ifd_off, endian).expect("ifd parse");
    let tables = info.jpeg_tables.expect("JPEGTables must be present");
    assert!(tables.len() > 4, "JPEGTables payload too small");
    // The abbreviated stream begins with SOI and ends with EOI.
    assert_eq!(&tables[0..2], &[0xFF, 0xD8]);
    assert_eq!(&tables[tables.len() - 2..], &[0xFF, 0xD9]);

    let _ = std::fs::remove_file(&path);
}

// -----------------------------------------------------------------------
// CCITT fax compression round trips (compress_image_data → decompress_strip)
// -----------------------------------------------------------------------

/// Pack a bilevel test pattern into a 1-bpp bitmap (WhiteIsZero polarity,
/// 1 bit = black).
fn pack_bilevel(width: usize, height: usize, black: impl Fn(usize, usize) -> bool) -> Vec<u8> {
    let row_bytes = width.div_ceil(8);
    let mut out = vec![0u8; row_bytes * height];
    for y in 0..height {
        for x in 0..width {
            if black(x, y) {
                out[y * row_bytes + (x >> 3)] |= 1 << (7 - (x & 7));
            }
        }
    }
    out
}

fn bilevel_info(width: u32, height: u32, compression: TiffCompression) -> TiffInfo {
    let mut info = TiffInfo::default();
    info.width = width;
    info.height = height;
    info.bits_per_sample = vec![1];
    info.samples_per_pixel = 1;
    info.compression = compression;
    info.photometric = PhotometricInterpretation::WhiteIsZero;
    info.rows_per_strip = height;
    info
}

#[test]
fn test_ccitt_rle_tiff_codec_roundtrip() {
    let width = 40u32;
    let height = 12u32;
    let packed = pack_bilevel(width as usize, height as usize, |x, y| (x / 3 + y) % 2 == 0);
    let info = bilevel_info(width, height, TiffCompression::CcittRle);
    let compressed = compress_image_data(&packed, &info).expect("ccitt rle encode");
    let decoded = decompress_strip(&compressed, &info, height as usize).expect("ccitt rle decode");
    assert_eq!(decoded, packed, "CcittRle TIFF codec must round-trip");
}

#[test]
fn test_ccitt_g3_tiff_codec_roundtrip() {
    let width = 96u32;
    let height = 20u32;
    let packed = pack_bilevel(width as usize, height as usize, |x, y| {
        ((x as i32 - y as i32 * 2).rem_euclid(20)) < 11
    });
    let info = bilevel_info(width, height, TiffCompression::CcittFax3);
    let compressed = compress_image_data(&packed, &info).expect("g3 encode");
    let decoded = decompress_strip(&compressed, &info, height as usize).expect("g3 decode");
    assert_eq!(decoded, packed, "CcittFax3 TIFF codec must round-trip");
}

#[test]
fn test_ccitt_g4_tiff_codec_roundtrip() {
    let width = 128u32;
    let height = 64u32;
    let packed = pack_bilevel(width as usize, height as usize, |x, y| {
        let cx = x as i32 - 64;
        let cy = y as i32 - 32;
        cx * cx + cy * cy * 4 < 1600
    });
    let info = bilevel_info(width, height, TiffCompression::CcittFax4);
    let compressed = compress_image_data(&packed, &info).expect("g4 encode");
    // G4 should compress this filled-ellipse image well below raw size.
    assert!(
        compressed.len() < packed.len(),
        "G4 should compress: {} vs {}",
        compressed.len(),
        packed.len()
    );
    let decoded = decompress_strip(&compressed, &info, height as usize).expect("g4 decode");
    assert_eq!(decoded, packed, "CcittFax4 TIFF codec must round-trip");
}

/// Build a complete single-strip bilevel TIFF (little-endian) with the
/// given compression and pre-encoded strip bytes.
fn build_bilevel_tiff(width: u32, height: u32, compression: u16, strip: &[u8]) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.write_u16::<LittleEndian>(TIFF_MAGIC_LE).expect("magic");
    buf.write_u16::<LittleEndian>(TIFF_VERSION).expect("ver");
    buf.write_u32::<LittleEndian>(0).expect("ifd ptr");

    // Strip data immediately after the 8-byte header.
    let strip_offset = buf.len() as u32;
    buf.extend_from_slice(strip);

    let ifd_pos = buf.len() as u32;
    buf.write_u16::<LittleEndian>(8).expect("tag count");

    macro_rules! tag {
        ($tag:expr, $dtype:expr, $count:expr, $val:expr) => {{
            buf.write_u16::<LittleEndian>($tag).expect("tag");
            buf.write_u16::<LittleEndian>($dtype).expect("dtype");
            buf.write_u32::<LittleEndian>($count).expect("count");
            buf.write_u32::<LittleEndian>($val).expect("value");
        }};
    }
    tag!(256, 4, 1, width); // ImageWidth
    tag!(257, 4, 1, height); // ImageLength
    tag!(258, 3, 1, 1); // BitsPerSample = 1
    tag!(259, 3, 1, u32::from(compression)); // Compression
    tag!(262, 3, 1, 0); // PhotometricInterpretation = WhiteIsZero
    tag!(273, 4, 1, strip_offset); // StripOffsets
    tag!(278, 4, 1, height); // RowsPerStrip
    tag!(279, 4, 1, strip.len() as u32); // StripByteCounts
    buf.write_u32::<LittleEndian>(0).expect("next ifd");

    buf[4..8].copy_from_slice(&ifd_pos.to_le_bytes());
    buf
}

#[test]
fn test_ccitt_g4_tiff_file_read() {
    // Encode a known bitmap, embed it in a hand-built TIFF, read it back
    // through the full read_tiff path.
    let width = 64u32;
    let height = 32u32;
    let packed = pack_bilevel(width as usize, height as usize, |x, y| (x + y) % 5 < 2);
    let strip = ccitt::encode_ccitt_fax4(&packed, width as usize, height as usize, true);
    let tiff_bytes = build_bilevel_tiff(width, height, 4, &strip);

    let mut path = std::env::temp_dir();
    path.push("oximedia_tiff_ccitt_g4_file.tif");
    std::fs::write(&path, &tiff_bytes).expect("write tiff");

    let frame = read_tiff(&path, 0).expect("read CCITT G4 TIFF");
    assert_eq!(frame.width, width);
    assert_eq!(frame.height, height);
    let data = frame.data.as_slice().expect("interleaved");
    assert_eq!(data, packed.as_slice(), "G4 TIFF file must decode exactly");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_ccitt_g3_tiff_file_read() {
    let width = 80u32;
    let height = 24u32;
    let packed = pack_bilevel(width as usize, height as usize, |x, _| (x / 7) % 2 == 1);
    let strip = ccitt::encode_ccitt_fax3(&packed, width as usize, height as usize, true);
    let tiff_bytes = build_bilevel_tiff(width, height, 3, &strip);

    let mut path = std::env::temp_dir();
    path.push("oximedia_tiff_ccitt_g3_file.tif");
    std::fs::write(&path, &tiff_bytes).expect("write tiff");

    let frame = read_tiff(&path, 0).expect("read CCITT G3 TIFF");
    let data = frame.data.as_slice().expect("interleaved");
    assert_eq!(data, packed.as_slice(), "G3 TIFF file must decode exactly");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_old_style_jpeg_rejected() {
    // Compression 6 (old-style JPEG) must be rejected, not silently
    // mis-decoded.
    let mut info = TiffInfo::default();
    info.width = 8;
    info.height = 8;
    info.compression = TiffCompression::JpegOld;
    let err = decompress_strip(&[0u8; 16], &info, 8);
    assert!(err.is_err(), "old-style JPEG must be rejected on read");

    let frame = ImageFrame::new(
        0,
        8,
        8,
        PixelType::U8,
        1,
        ColorSpace::Luma,
        ImageData::interleaved(vec![0u8; 64]),
    );
    let info2 = create_tiff_info(&frame, TiffCompression::JpegOld).expect("info");
    let werr = compress_image_data(&vec![0u8; 64], &info2);
    assert!(werr.is_err(), "old-style JPEG must be rejected on write");
}

/// Build a minimal TIFF byte buffer in memory (little-endian) with a single
/// extra IFD entry appended after the standard tags.  Used by the round-trip
/// tests to inject an unknown tag without going through the full write path.
fn build_tiff_with_extra_tag(
    width: u32,
    height: u32,
    extra_tag: u16,
    extra_dtype: u16,
    extra_count: u32,
    extra_value: u32,
) -> Vec<u8> {
    use byteorder::{LittleEndian, WriteBytesExt};

    // Pixel data: `width * height` bytes of 0x80 (gray).
    let pixel_data: Vec<u8> = vec![0x80u8; (width * height) as usize];

    let mut buf: Vec<u8> = Vec::new();
    // Header: II + 42 + placeholder IFD offset
    buf.write_u16::<LittleEndian>(0x4949).expect("magic");
    buf.write_u16::<LittleEndian>(42).expect("version");
    buf.write_u32::<LittleEndian>(0)
        .expect("ifd ptr placeholder");

    let strip_offset = buf.len() as u32;
    buf.extend_from_slice(&pixel_data);

    let ifd_pos = buf.len() as u32;
    // 9 standard tags + 1 extra tag
    let tag_count: u16 = 9 + 1;
    buf.write_u16::<LittleEndian>(tag_count).expect("tag count");

    macro_rules! tag {
        ($t:expr, $dt:expr, $cnt:expr, $val:expr) => {{
            buf.write_u16::<LittleEndian>($t).expect("tag");
            buf.write_u16::<LittleEndian>($dt).expect("dtype");
            buf.write_u32::<LittleEndian>($cnt).expect("count");
            buf.write_u32::<LittleEndian>($val).expect("value");
        }};
    }
    tag!(256, 4, 1, width); // ImageWidth
    tag!(257, 4, 1, height); // ImageLength
    tag!(258, 3, 1, 0x0008); // BitsPerSample = 8
    tag!(259, 3, 1, 1u32); // Compression = None
    tag!(262, 3, 1, 1u32); // PhotometricInterpretation = BlackIsZero
    tag!(273, 4, 1, u32::from(strip_offset)); // StripOffsets
    tag!(277, 3, 1, 0x0001); // SamplesPerPixel = 1
    tag!(278, 4, 1, height); // RowsPerStrip
    tag!(279, 4, 1, pixel_data.len() as u32); // StripByteCounts
                                              // Extra unknown tag
    tag!(extra_tag, extra_dtype, extra_count, extra_value);

    buf.write_u32::<LittleEndian>(0).expect("next ifd = 0");

    // Patch IFD offset in header
    buf[4..8].copy_from_slice(&ifd_pos.to_le_bytes());
    buf
}

#[test]
fn test_tiff_round_trip_preserves_unknown_tag_0xc4a5() {
    // Tag 0xC4A5 (50341) = Photoshop private tag.  Value = 42u16 packed as Short.
    let custom_tag: u16 = 0xC4A5;
    let custom_value: u32 = 42;
    let tiff_bytes =
        build_tiff_with_extra_tag(8, 8, custom_tag, 3 /* Short */, 1, custom_value);

    let mut path = std::env::temp_dir();
    path.push("oximedia_tiff_extra_tag_c4a5.tif");
    std::fs::write(&path, &tiff_bytes).expect("write TIFF");

    // Use read_ifd directly to inspect the preserved extra tags.
    let mut file = std::fs::File::open(&path).expect("open");
    use byteorder::ReadBytesExt;
    let _magic = file.read_u16::<byteorder::LittleEndian>().expect("magic");
    let _ver = file.read_u16::<byteorder::LittleEndian>().expect("ver");
    let ifd_off = u64::from(file.read_u32::<byteorder::LittleEndian>().expect("ifd off"));
    let info = read_ifd(&mut file, ifd_off, Endian::Little).expect("read_ifd");

    let entry = info
        .get_tag(custom_tag)
        .unwrap_or_else(|| panic!("custom tag 0x{custom_tag:04X} must be preserved"));
    // The value_offset field holds the inline short value = 42.
    assert_eq!(
        entry.value_offset & 0xFFFF,
        custom_value,
        "custom tag value must be preserved"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_tiff_round_trip_preserves_gps_ifd_0x8825() {
    // Tag 0x8825 (34853) = GPS IFD pointer.
    let gps_tag: u16 = 0x8825;
    let gps_ptr: u32 = 0x0000_0100; // dummy pointer value
    let tiff_bytes = build_tiff_with_extra_tag(8, 8, gps_tag, 4 /* Long */, 1, gps_ptr);

    let mut path = std::env::temp_dir();
    path.push("oximedia_tiff_extra_tag_gps.tif");
    std::fs::write(&path, &tiff_bytes).expect("write TIFF");

    let mut file = std::fs::File::open(&path).expect("open");
    use byteorder::ReadBytesExt;
    let _magic = file.read_u16::<byteorder::LittleEndian>().expect("magic");
    let _ver = file.read_u16::<byteorder::LittleEndian>().expect("ver");
    let ifd_off = u64::from(file.read_u32::<byteorder::LittleEndian>().expect("ifd off"));
    let info = read_ifd(&mut file, ifd_off, Endian::Little).expect("read_ifd");

    let entry = info
        .get_tag(gps_tag)
        .unwrap_or_else(|| panic!("GPS IFD tag 0x{gps_tag:04X} must be preserved"));
    assert_eq!(
        entry.value_offset, gps_ptr,
        "GPS IFD pointer must be preserved verbatim"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_tiff_set_get_tag() {
    let mut info = TiffInfo::default();
    let entry = IfdEntry {
        tag: 0xBEEF,
        data_type: TiffDataType::Short,
        count: 1,
        value_offset: 999,
    };
    info.set_tag(0xBEEF, entry);
    let retrieved = info.get_tag(0xBEEF).expect("tag must be present");
    assert_eq!(retrieved.value_offset, 999);
    assert!(info.get_tag(0xDEAD).is_none());
}
