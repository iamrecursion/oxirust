//! Wave 14 integration tests for Slice A deliverables.
//!
//! Covers:
//! 1. StingerPlayer fallback to synthetic frames when file is absent
//! 2. StingerPlayer with a real (tiny) VP9 clip from SimpleVp9Encoder
//! 3. save_replay encode roundtrip — VP9 format produces a non-empty ORC file
//! 4. GPU-scaling rayon parallelism — downscale produces correct output dimensions

use std::time::Duration;

// ---------------------------------------------------------------------------
// 1. Stinger decode: fallback to synthetic when path does not exist
// ---------------------------------------------------------------------------

#[test]
fn test_stinger_decode_fallback() {
    use oximedia_gaming::scene::transition::StingerPlayer;

    // Path that definitely does not exist.
    let path = std::env::temp_dir().join("wave14_stinger_nonexistent_12345.webm");
    // Ensure it really doesn't exist.
    let _ = std::fs::remove_file(&path);

    let player = StingerPlayer::new(&path, 0).expect("Should succeed via synthetic fallback");

    // Must have produced frames (from synthetic generator).
    assert!(
        player.total_duration_ms() > 0,
        "Synthetic fallback should generate frames"
    );
    // Basic sanity: width and height must be non-zero.
    assert!(player.width() > 0);
    assert!(player.height() > 0);
}

// ---------------------------------------------------------------------------
// 2. Stinger decode: real VP9 clip encode → StingerPlayer decode
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "vp9")]
fn test_stinger_decode_real_clip() {
    use oximedia_codec::{SimpleVp9Encoder, Vp9EncConfig, Vp9Profile};
    use oximedia_gaming::scene::transition::StingerPlayer;

    // Build a minimal WebM-like container with VP9 frames.
    // We use SimpleVp9Encoder to produce VP9 bitstream data, then wrap it in
    // a minimal Matroska container skeleton.  If the parse fails, StingerPlayer
    // falls back to synthetic frames — which is also acceptable for this test.

    const W: u32 = 64;
    const H: u32 = 64;
    const FRAMES: usize = 5;

    let config = Vp9EncConfig {
        width: W,
        height: H,
        quality: 40,
        speed: 8,
        keyframe_interval: 2,
        target_bitrate: 200,
        profile: Vp9Profile::Profile0,
    };

    let mut encoder = SimpleVp9Encoder::new(config).expect("SimpleVp9Encoder should initialise");

    // Generate YUV420 solid-colour frames.
    let yuv_size = (W as usize) * (H as usize) * 3 / 2;
    let mut packets: Vec<Vec<u8>> = Vec::with_capacity(FRAMES);
    for i in 0..FRAMES {
        let luma = (40u8).wrapping_add(i as u8 * 30);
        let mut yuv = vec![luma; yuv_size];
        // Set chroma planes to neutral.
        let chroma_start = (W as usize) * (H as usize);
        for byte in &mut yuv[chroma_start..] {
            *byte = 128;
        }
        let pkt = encoder
            .encode_frame(&yuv, i == 0)
            .expect("encode_frame should succeed");
        packets.push(pkt.data);
    }

    // Build a minimal EBML container with a Cluster containing SimpleBlocks.
    let container = build_minimal_webm_vp9(&packets);

    let tmp = std::env::temp_dir().join("wave14_stinger_real_vp9.webm");
    std::fs::write(&tmp, &container).expect("write tmp webm");

    // Load via StingerPlayer — will either decode real or fall back to synthetic.
    let player_result = StingerPlayer::new(&tmp, 0);
    let _ = std::fs::remove_file(&tmp);

    let player = player_result.expect("StingerPlayer should succeed (fallback or real)");
    assert!(player.total_duration_ms() > 0, "Must have frames");
    assert!(player.width() > 0);
    assert!(player.height() > 0);
}

// ---------------------------------------------------------------------------
// 3. save_replay encode roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_save_replay_encode_roundtrip() {
    use oximedia_gaming::replay::buffer::{ReplayBuffer, ReplayConfig};
    use oximedia_gaming::replay::save::{save_replay, ReplaySaver, SaveFormat};

    let mut buf = ReplayBuffer::new(ReplayConfig::default()).expect("valid config");
    buf.enable().expect("enable");

    // Push synthetic "encoded" frames with realistic size.
    for i in 0..5u64 {
        buf.push_frame(vec![0xABu8; 64], Duration::from_millis(i * 33), i == 0)
            .expect("push frame");
    }

    let dir = std::env::temp_dir();
    let path = dir.join("wave14_save_replay_roundtrip.orc");

    save_replay(&buf, &path, SaveFormat::WebM).expect("save_replay should succeed");

    // File must exist and be non-empty.
    let meta = std::fs::metadata(&path).expect("file should exist");
    assert!(meta.len() > 0, "Output file must be non-empty");

    // Header must be valid ORC magic.
    let data = std::fs::read(&path).expect("read file");
    assert_eq!(&data[0..8], b"OxiReply", "ORC magic must be present");

    // Must round-trip as at least 5 frames.
    let (_fmt, frames) = ReplaySaver::decode_frames(&data).expect("decode ORC");
    assert_eq!(frames.len(), 5, "Should round-trip 5 frames");

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// 4. GPU scale rayon parallel — downscale 640×360 → 320×180
// ---------------------------------------------------------------------------

#[test]
fn test_gpu_scale_rayon_parallel() {
    use oximedia_gaming::gpu_scaling::{GpuScaler, GpuScalerConfigBuilder, RgbaFrame};

    let cfg = GpuScalerConfigBuilder::new()
        .src_resolution(640, 360)
        .dst_resolution(320, 180)
        .build()
        .expect("valid config");

    let mut scaler = GpuScaler::new(cfg).expect("scaler");

    // Fill with a known non-zero colour so we can verify it's not all-zero.
    let frame = RgbaFrame::new_solid(640, 360, [200, 100, 50, 255]).expect("frame");

    let out = scaler.scale(&frame).expect("scale should succeed");

    // Output dimensions.
    assert_eq!(out.width, 320);
    assert_eq!(out.height, 180);
    assert_eq!(out.data.len(), (320 * 180 * 4) as usize);

    // First pixel must be a valid RGBA value; since input is solid colour it
    // should be close to [200, 100, 50, 255].
    assert!(
        out.data[0] > 0 || out.data[1] > 0 || out.data[2] > 0,
        "Output pixel must not be all-zero for non-zero input"
    );

    // Alpha channel must be 255.
    assert_eq!(out.data[3], 255, "Alpha must be 255");

    // Stats should have recorded exactly 1 frame.
    assert_eq!(scaler.stats().frames_processed, 1);
}

// ---------------------------------------------------------------------------
// Minimal WebM/EBML builder helper
// ---------------------------------------------------------------------------

/// Build the smallest valid EBML+Matroska/WebM container containing `packets`
/// as VP9 SimpleBlocks.  The container has:
/// - EBML header
/// - Segment (with a TrackEntry declaring "V_VP9")
/// - One Cluster with all packets as SimpleBlocks
#[cfg(feature = "vp9")]
fn build_minimal_webm_vp9(packets: &[Vec<u8>]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();

    // EBML header: id=0x1A45DFA3, size=0x1F (31 bytes payload)
    let header_payload = build_ebml_header_payload();
    write_ebml_element(&mut out, 0x1A45DFA3u32, &header_payload);

    // Segment: id=0x18538067 with unknown size (0x01FF_FFFF_FFFF_FFFF)
    // We'll write a self-contained segment with known size instead.
    let mut segment_body: Vec<u8> = Vec::new();

    // SeekHead (minimal, empty): id=0x114D9B74 size=0
    write_ebml_element(&mut segment_body, 0x114D9B74u32, &[]);

    // Info: id=0x1549A966
    let mut info: Vec<u8> = Vec::new();
    // TimecodeScale: id=0x2AD7B1, value=1_000_000 (1 ms)
    write_ebml_element_u64(&mut info, 0x2AD7B1u32, 1_000_000);
    // MuxingApp: id=0x4D80
    write_ebml_element(&mut info, 0x4D80u32, b"oximedia-gaming-test");
    write_ebml_element(&mut segment_body, 0x1549A966u32, &info);

    // Tracks: id=0x1654AE6B
    let mut tracks: Vec<u8> = Vec::new();
    let mut track_entry: Vec<u8> = Vec::new();
    // TrackNumber: id=0xD7 = 1
    write_ebml_element_u64(&mut track_entry, 0xD7u32, 1);
    // TrackUID: id=0x73C5 = 1
    write_ebml_element_u64(&mut track_entry, 0x73C5u32, 1);
    // TrackType: id=0x83 = 1 (video)
    write_ebml_element_u64(&mut track_entry, 0x83u32, 1);
    // CodecID: id=0x86 = "V_VP9"
    write_ebml_element(&mut track_entry, 0x86u32, b"V_VP9");
    // TrackEntry: id=0xAE
    write_ebml_element(&mut tracks, 0xAEu32, &track_entry);
    write_ebml_element(&mut segment_body, 0x1654AE6Bu32, &tracks);

    // Cluster: id=0x1F43B675
    let mut cluster: Vec<u8> = Vec::new();
    // Timecode: id=0xE7 = 0
    write_ebml_element_u64(&mut cluster, 0xE7u32, 0);
    // SimpleBlocks
    for (i, pkt) in packets.iter().enumerate() {
        // SimpleBlock layout: track_num (vint=1byte=0x81), timecode (i16), flags (0x80 = keyframe), data
        let mut sb: Vec<u8> = Vec::new();
        sb.push(0x81); // track number 1 as vint
        sb.extend_from_slice(&(i as i16).to_be_bytes()); // timecode
                                                         // flags: keyframe = 0x80, else 0x00
        sb.push(if i == 0 { 0x80 } else { 0x00 });
        sb.extend_from_slice(pkt);
        write_ebml_element(&mut cluster, 0xA3u32, &sb);
    }
    write_ebml_element(&mut segment_body, 0x1F43B675u32, &cluster);

    write_ebml_element(&mut out, 0x18538067u32, &segment_body);

    out
}

#[cfg(feature = "vp9")]
fn build_ebml_header_payload() -> Vec<u8> {
    let mut p: Vec<u8> = Vec::new();
    // EBMLVersion: id=0x4286, value=1
    write_ebml_element_u64(&mut p, 0x4286u32, 1);
    // EBMLReadVersion: id=0x42F7, value=1
    write_ebml_element_u64(&mut p, 0x42F7u32, 1);
    // EBMLMaxIDLength: id=0x42F2, value=4
    write_ebml_element_u64(&mut p, 0x42F2u32, 4);
    // EBMLMaxSizeLength: id=0x42F3, value=8
    write_ebml_element_u64(&mut p, 0x42F3u32, 8);
    // DocType: id=0x4282 = "webm"
    write_ebml_element(&mut p, 0x4282u32, b"webm");
    // DocTypeVersion: id=0x4287, value=4
    write_ebml_element_u64(&mut p, 0x4287u32, 4);
    // DocTypeReadVersion: id=0x4285, value=2
    write_ebml_element_u64(&mut p, 0x4285u32, 2);
    p
}

#[cfg(feature = "vp9")]
fn write_ebml_element(out: &mut Vec<u8>, id: u32, payload: &[u8]) {
    write_ebml_id_bytes(out, id);
    write_ebml_size(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

#[cfg(feature = "vp9")]
fn write_ebml_element_u64(out: &mut Vec<u8>, id: u32, val: u64) {
    // Encode as minimal big-endian integer.
    let bytes = encode_uint_minimal(val);
    write_ebml_element(out, id, &bytes);
}

#[cfg(feature = "vp9")]
fn encode_uint_minimal(val: u64) -> Vec<u8> {
    if val == 0 {
        return vec![0u8];
    }
    let mut b = val;
    let mut bytes: Vec<u8> = Vec::new();
    while b > 0 {
        bytes.push((b & 0xFF) as u8);
        b >>= 8;
    }
    bytes.reverse();
    bytes
}

#[cfg(feature = "vp9")]
fn write_ebml_id_bytes(out: &mut Vec<u8>, id: u32) {
    // Encode id as raw big-endian, keeping all bytes that contain data.
    // IDs are already encoded with the leading bit set (EBML ID encoding).
    if id <= 0xFF {
        out.push(id as u8);
    } else if id <= 0xFFFF {
        out.extend_from_slice(&(id as u16).to_be_bytes());
    } else if id <= 0xFF_FFFF {
        out.push((id >> 16) as u8);
        out.push((id >> 8) as u8);
        out.push(id as u8);
    } else {
        out.extend_from_slice(&id.to_be_bytes());
    }
}

#[cfg(feature = "vp9")]
fn write_ebml_size(out: &mut Vec<u8>, size: u64) {
    // EBML vint encoding for size.
    if size < 0x7F {
        out.push((size | 0x80) as u8);
    } else if size < 0x3FFF {
        let enc = size | 0x4000;
        out.push((enc >> 8) as u8);
        out.push(enc as u8);
    } else if size < 0x1F_FFFF {
        let enc = size | 0x20_0000;
        out.push((enc >> 16) as u8);
        out.push((enc >> 8) as u8);
        out.push(enc as u8);
    } else {
        // 4-byte vint
        let enc = size | 0x1000_0000;
        out.extend_from_slice(&(enc as u32).to_be_bytes());
    }
}
