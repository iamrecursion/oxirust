//! Malformed-file rejection tests: `Hdt::from_bytes` must reject corrupted input with a
//! diagnosable [`HdtError`], never panic or silently return wrong data.

use oxixml_hdt::{Hdt, HdtError, HdtWriter};
use oxixml_model::{NamedNode, Triple};

/// Builds a small, valid HDT file to corrupt in each test.
fn valid_hdt_bytes() -> Vec<u8> {
    let ex = |s: &str| NamedNode::new(format!("http://example.com/{s}")).expect("valid iri");
    let triples = vec![
        Triple::new(ex("a"), ex("knows"), ex("b")),
        Triple::new(ex("b"), ex("knows"), ex("c")),
        Triple::new(ex("c"), ex("knows"), ex("a")),
    ];
    let mut bytes = Vec::new();
    HdtWriter::new()
        .write(&mut bytes, triples)
        .expect("write should succeed");
    bytes
}

#[test]
fn rejects_empty_input() {
    let error = Hdt::from_bytes(&[]).expect_err("empty input is not a valid HDT file");
    assert!(matches!(error, HdtError::UnexpectedEof { .. }));
}

#[test]
fn rejects_missing_cookie() {
    let bytes = b"this is definitely not an HDT file".to_vec();
    let error = Hdt::from_bytes(&bytes).expect_err("garbage input must be rejected");
    assert!(matches!(error, HdtError::InvalidCookie { .. }));
}

#[test]
fn rejects_truncated_file_at_every_prefix_length() {
    let bytes = valid_hdt_bytes();
    // Every strict prefix of a valid file must fail to parse (never panic, never succeed).
    for length in 0..bytes.len() {
        let prefix = &bytes[..length];
        assert!(
            Hdt::from_bytes(prefix).is_err(),
            "a {length}-byte prefix of a valid file must not parse successfully"
        );
    }
}

#[test]
fn rejects_corrupted_global_control_info_crc() {
    let mut bytes = valid_hdt_bytes();
    // Flip a bit inside the global control information's format string, which is covered by
    // its CRC16 but comes before any section-specific CRC.
    bytes[5] ^= 0xFF;
    let error = Hdt::from_bytes(&bytes).expect_err("corrupted CRC16 must be detected");
    assert!(matches!(error, HdtError::CrcMismatch { kind: "CRC16", .. }));
}

#[test]
fn rejects_corrupted_dictionary_buffer() {
    let bytes = valid_hdt_bytes();
    // Flip a byte roughly in the middle of the file, which (for this small fixture) lands
    // inside the dictionary section. Depending on the exact offset this is caught either by a
    // CRC (inside a framed sub-structure) or by an outright unrecognized section-type marker
    // (the un-checksummed 4-byte type prefix that precedes each dictionary section) -- either
    // way, `Hdt::from_bytes` must reject it cleanly rather than returning wrong data.
    let mut corrupted = bytes.clone();
    let middle = bytes.len() / 2;
    corrupted[middle] ^= 0xFF;
    Hdt::from_bytes(&corrupted).expect_err("corrupted payload must be detected");
}

#[test]
fn rejects_corrupted_trailing_bytes() {
    let bytes = valid_hdt_bytes();
    let mut corrupted = bytes.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 0xFF;
    let error = Hdt::from_bytes(&corrupted).expect_err("corrupted trailing CRC must be detected");
    assert!(matches!(error, HdtError::CrcMismatch { .. }));
}

#[test]
fn rejects_wrong_cookie_bytes_but_right_length() {
    let mut bytes = valid_hdt_bytes();
    bytes[0] = b'X';
    let error = Hdt::from_bytes(&bytes).expect_err("wrong cookie must be rejected");
    assert!(matches!(error, HdtError::InvalidCookie { .. }));
}

#[test]
fn every_single_byte_flip_is_either_rejected_or_still_correct() {
    // A stronger property: for a sample of byte offsets, flipping a single bit must never
    // cause `Hdt::from_bytes` to panic, and if it *does* still parse (extremely unlikely, but
    // not impossible for e.g. flipping a byte in an unrelated property string), it must not
    // silently corrupt the triple count.
    let bytes = valid_hdt_bytes();
    let original = Hdt::from_bytes(&bytes).expect("baseline parses");
    for offset in (0..bytes.len()).step_by(7) {
        let mut corrupted = bytes.clone();
        corrupted[offset] ^= 0x01;
        match Hdt::from_bytes(&corrupted) {
            Err(_) => {}
            Ok(hdt) => assert_eq!(hdt.len(), original.len()),
        }
    }
}
