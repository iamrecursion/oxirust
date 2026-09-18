//! Corrupt / malformed Structured-Storage input tests.
//!
//! Verifies that `StorageReader::new` rejects garbage, truncated, empty, and
//! bad-version compound-file headers with a graceful `Err` instead of panicking.
//!
//! Byte layout of the compound-file header (from `structured_storage::Header::parse`):
//!   offset  0: 8-byte signature `D0 CF 11 E0 A1 B1 1A E1`
//!   offset  8: 16-byte CLSID (skipped)
//!   offset 24: u16 LE minor version
//!   offset 26: u16 LE major version  (must be 3 or 4)
//!   offset 28: u16 LE byte order     (must be 0xFFFE)
//! The version check at offset 26 runs BEFORE the byte-order check at offset 28.

use oximedia_aaf::structured_storage::StorageReader;
use oximedia_aaf::AafError;
use std::io::Cursor;

/// 4 KiB of `0xAA` has a bad signature → `InvalidStructuredStorage`, no panic.
#[test]
fn garbage_bytes_error_not_panic() {
    let r = StorageReader::new(Cursor::new(vec![0xAAu8; 4096]));
    assert!(
        matches!(r.err(), Some(AafError::InvalidStructuredStorage(_))),
        "garbage bytes must surface as InvalidStructuredStorage"
    );
}

/// Input shorter than the 8-byte signature cannot satisfy `read_exact`, so the
/// reader errors (an `Io` underflow or `InvalidStructuredStorage`) — never panics.
#[test]
fn truncated_below_signature_errors() {
    let r = StorageReader::new(Cursor::new(vec![0xD0u8, 0xCF, 0x11, 0xE0]));
    assert!(
        r.is_err(),
        "input truncated below the signature length must error"
    );
    // Assert the error is one of the two graceful variants we expect.
    match r.err() {
        Some(AafError::Io(_)) | Some(AafError::InvalidStructuredStorage(_)) => {}
        other => panic!("expected Io or InvalidStructuredStorage, got {other:?}"),
    }
}

/// Empty input must error gracefully (read_exact on 0 bytes fails) — no panic.
#[test]
fn empty_input_errors_gracefully() {
    let r = StorageReader::new(Cursor::new(Vec::<u8>::new()));
    assert!(r.is_err(), "empty input must error, not panic");
}

/// A valid signature followed by an unsupported major version (9) must be
/// rejected with `InvalidStructuredStorage` whose message names the version.
#[test]
fn valid_magic_bad_version_errors() {
    let mut buf: Vec<u8> = Vec::with_capacity(80);
    // 8-byte signature.
    buf.extend_from_slice(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1");
    // 16-byte CLSID (zeros).
    buf.extend_from_slice(&[0u8; 16]);
    // Minor version (2 bytes LE).
    buf.extend_from_slice(&0u16.to_le_bytes());
    // Major version 0x0009 (2 bytes LE) — unsupported.
    buf.extend_from_slice(&0x0009u16.to_le_bytes());
    // Zero-pad to >= 80 bytes so any further reads have data available.
    buf.resize(80, 0);

    let r = StorageReader::new(Cursor::new(buf));
    match r.err() {
        Some(AafError::InvalidStructuredStorage(msg)) => {
            assert!(
                msg.contains("Unsupported version"),
                "version error message must mention 'Unsupported version', got: {msg}"
            );
        }
        other => panic!("expected InvalidStructuredStorage(Unsupported version), got {other:?}"),
    }
}
