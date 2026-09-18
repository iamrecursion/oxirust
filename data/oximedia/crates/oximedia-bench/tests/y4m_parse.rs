//! Tests for Y4M (YUV4MPEG2) sequence parsing.

use oximedia_bench::sequences::{Y4mError, Y4mSequence};
use std::io::Write;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build a minimal Y4M byte stream in memory.
///
/// `frames` is a slice of `(y_plane, u_plane, v_plane)` for each frame.
fn make_y4m(width: u32, height: u32, chroma: &str, frames: &[Vec<u8>]) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = writeln!(
        buf,
        "YUV4MPEG2 W{} H{} F25:1 Ip A1:1 C{}",
        width, height, chroma
    );
    for frame_data in frames {
        let _ = writeln!(buf, "FRAME");
        buf.extend_from_slice(frame_data);
    }
    buf
}

fn frame_420(width: u32, height: u32, fill: u8) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let y = vec![fill; w * h];
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    let mut d = y;
    d.extend_from_slice(&u);
    d.extend_from_slice(&v);
    d
}

// ── round-trip tests ──────────────────────────────────────────────────────────

#[test]
fn test_round_trip_4_frames_64x64() {
    let frames: Vec<Vec<u8>> = (0..4).map(|i| frame_420(64, 64, i as u8 * 10)).collect();
    let raw = make_y4m(64, 64, "420", &frames);

    let mut seq = Y4mSequence::from_reader(raw.as_slice()).expect("parse header");
    assert_eq!(seq.width(), 64);
    assert_eq!(seq.height(), 64);
    assert_eq!(seq.chroma(), "420");

    let mut count = 0u64;
    while let Some(frame) = seq.next_frame().expect("read frame") {
        assert_eq!(frame.pts, count);
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        // Verify the Y plane fill value matches what we wrote.
        let expected_fill = (count as u8) * 10;
        assert!(
            frame.plane_y.iter().all(|&b| b == expected_fill),
            "frame {count} Y plane mismatch"
        );
        assert_eq!(frame.plane_y.len(), 64 * 64);
        assert_eq!(frame.plane_u.len(), 32 * 32);
        assert_eq!(frame.plane_v.len(), 32 * 32);
        count += 1;
    }
    assert_eq!(count, 4, "expected 4 frames");
}

#[test]
fn test_444_chroma() {
    let w = 16u32;
    let h = 16u32;
    let frame_data = {
        let y = vec![100u8; (w * h) as usize];
        let u = vec![50u8; (w * h) as usize];
        let v = vec![200u8; (w * h) as usize];
        let mut d = y;
        d.extend_from_slice(&u);
        d.extend_from_slice(&v);
        d
    };
    let raw = make_y4m(w, h, "444", &[frame_data]);
    let mut seq = Y4mSequence::from_reader(raw.as_slice()).expect("parse header");
    assert_eq!(seq.chroma(), "444");
    let frame = seq.next_frame().expect("read").expect("frame");
    assert_eq!(frame.plane_y.len(), (w * h) as usize);
    assert_eq!(frame.plane_u.len(), (w * h) as usize);
    assert_eq!(frame.plane_v.len(), (w * h) as usize);
}

// ── error cases ───────────────────────────────────────────────────────────────

#[test]
fn test_reject_malformed_header_no_magic() {
    let bad = b"NOT_Y4M W64 H64\n".to_vec();
    let result = Y4mSequence::from_reader(bad.as_slice());
    assert!(matches!(result, Err(Y4mError::MissingMagic)));
}

#[test]
fn test_reject_missing_width() {
    let bad = b"YUV4MPEG2 H64 F25:1 Ip\n".to_vec();
    let result = Y4mSequence::from_reader(bad.as_slice());
    assert!(matches!(result, Err(Y4mError::MissingField { field: "W" })));
}

#[test]
fn test_reject_missing_height() {
    let bad = b"YUV4MPEG2 W64 F25:1 Ip\n".to_vec();
    let result = Y4mSequence::from_reader(bad.as_slice());
    assert!(matches!(result, Err(Y4mError::MissingField { field: "H" })));
}

#[test]
fn test_reject_frame_size_mismatch() {
    // Write only half the expected frame bytes.
    let mut buf = Vec::new();
    let _ = writeln!(buf, "YUV4MPEG2 W64 H64 F25:1 Ip A1:1 C420");
    let _ = writeln!(buf, "FRAME");
    // Y + U + V for 64x64 420 = 4096 + 1024 + 1024 = 6144
    // Write only 100 bytes.
    buf.extend_from_slice(&vec![0u8; 100]);

    let mut seq = Y4mSequence::from_reader(buf.as_slice()).expect("header ok");
    let result = seq.next_frame();
    assert!(
        matches!(result, Err(Y4mError::TruncatedFrame { .. })),
        "expected TruncatedFrame, got: {result:?}"
    );
}

#[test]
fn test_zero_frames_returns_none() {
    let raw = make_y4m(32, 32, "420", &[]);
    let mut seq = Y4mSequence::from_reader(raw.as_slice()).expect("header");
    let frame = seq.next_frame().expect("no error");
    assert!(frame.is_none(), "expected None for empty stream");
}
