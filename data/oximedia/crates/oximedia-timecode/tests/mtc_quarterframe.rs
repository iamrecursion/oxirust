//! MIDI Timecode (MTC) quarter-frame conformance tests.
//!
//! MTC transmits one timecode frame as eight quarter-frame messages, each
//! carrying a 4-bit nibble tagged with its piece index in the high nibble.
//! These tests pin the exact wire bytes and verify the receiver reassembles a
//! byte-identical timecode after a full eight-piece sequence.

use oximedia_timecode::midi_timecode::{MtcFrameRate, MtcQuarterFrame, MtcReceiver, MtcTimecode};

/// Pins the exact quarter-frame data bytes for `01:02:03:04` @ 25 fps.
///
/// Each byte is `ppppdddd`: piece index in the high nibble, 4 bits of timecode
/// payload in the low nibble. Piece 7 packs the rate code (1 for 25 fps) and the
/// high bit of hours, giving `0x72`.
#[test]
fn quarter_frame_piece_values() {
    let tc = MtcTimecode::new(1, 2, 3, 4, MtcFrameRate::Fps25);
    let expect = [0x04u8, 0x10, 0x23, 0x30, 0x42, 0x50, 0x61, 0x72];
    for p in 0..8u8 {
        assert_eq!(
            MtcQuarterFrame::encode_quarter(&tc, p),
            expect[p as usize],
            "piece {p}"
        );
    }
}

/// Encoding a timecode to eight quarter frames and feeding them to a receiver
/// must reconstruct the identical timecode, including frame rate.
#[test]
fn quarter_frame_roundtrip_identity() {
    let tc = MtcTimecode::new(1, 30, 45, 12, MtcFrameRate::Fps30);
    let mut rx = MtcReceiver::new();
    let mut out = None;
    for p in 0..8u8 {
        out = rx.process_message(MtcQuarterFrame::encode_quarter(&tc, p));
    }

    let got = out.expect("complete after 8");
    assert_eq!(
        (got.hours, got.minutes, got.seconds, got.frames),
        (1, 30, 45, 12)
    );
    assert_eq!(got.frame_rate, MtcFrameRate::Fps30);
}

/// After only seven pieces the sequence is incomplete: the last
/// `process_message` returns `None` and the receiver reports not-complete.
#[test]
fn fewer_than_eight_incomplete() {
    let tc = MtcTimecode::new(1, 30, 45, 12, MtcFrameRate::Fps30);
    let mut rx = MtcReceiver::new();
    let mut out = None;
    for p in 0..7u8 {
        out = rx.process_message(MtcQuarterFrame::encode_quarter(&tc, p));
    }

    assert!(out.is_none(), "7 pieces must not complete a timecode");
    assert!(!rx.is_complete());
}
