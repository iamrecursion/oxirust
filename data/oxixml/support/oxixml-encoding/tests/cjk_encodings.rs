//! Integration tests for the four legacy CJK multi-byte encodings (GBK, gb18030, Big5, EUC-KR),
//! exercised purely through the public API (`Encoding`, `Decoder`, `encode_from_utf8`) rather
//! than the crate-internal `cjk` module directly -- mirroring `tests/japanese_encodings.rs`'s
//! end-to-end wiring coverage for the three legacy Japanese multi-byte encodings.

use oxixml_encoding::{DecodeErrorKind, Encoding, encode_from_utf8};

fn decode_strict(encoding: Encoding, bytes: &[u8]) -> Result<String, oxixml_encoding::DecodeError> {
    let mut decoder = encoding.new_decoder();
    let mut out = String::new();
    decoder.decode_to_string(bytes, &mut out, true)?;
    Ok(out)
}

fn decode_lossy(encoding: Encoding, bytes: &[u8]) -> String {
    let mut decoder = encoding.new_decoder();
    let mut out = String::new();
    decoder.decode_to_string_lossy(bytes, &mut out, true);
    out
}

// --- GBK / gb18030 -----------------------------------------------------------------------------

#[test]
fn gbk_ascii_compatibility_via_public_api() {
    let decoded =
        decode_strict(Encoding::Gbk, b"Hello, World! 123").expect("ASCII must decode as-is");
    assert_eq!(decoded, "Hello, World! 123");
}

#[test]
fn gbk_round_trips_via_public_api() {
    let mut bytes = Vec::new();
    encode_from_utf8("中文测试", Encoding::Gbk, &mut bytes).expect("must encode");
    let decoded = decode_strict(Encoding::Gbk, &bytes).expect("must decode");
    assert_eq!(decoded, "中文测试");
}

#[test]
fn gbk_rejects_supplementary_plane_character_gb18030_accepts_via_public_api() {
    let mut dst = Vec::new();
    let err = encode_from_utf8("\u{20000}", Encoding::Gbk, &mut dst).expect_err("GBK must fail");
    assert_eq!(err.character, '\u{20000}');

    let mut dst = Vec::new();
    encode_from_utf8("\u{20000}", Encoding::Gb18030, &mut dst).expect("gb18030 must encode it");
    let decoded = decode_strict(Encoding::Gb18030, &dst).expect("must decode");
    assert_eq!(decoded, "\u{20000}");
}

#[test]
fn gb18030_four_byte_supplementary_plane_character_via_public_api() {
    // U+20000, a CJK Extension B ideograph -- gb18030's encoder must reach for the 4-byte
    // algorithmic ranges since this code point has no 2-byte representation.
    let mut bytes = Vec::new();
    encode_from_utf8("\u{20000}\u{2A6D6}", Encoding::Gb18030, &mut bytes).expect("must encode");
    assert_eq!(bytes.len(), 8, "two 4-byte sequences");
    let decoded = decode_strict(Encoding::Gb18030, &bytes).expect("must decode");
    assert_eq!(decoded, "\u{20000}\u{2A6D6}");
}

#[test]
fn gb18030_truncated_four_byte_sequence_is_truncated_error_via_public_api() {
    let mut bytes = Vec::new();
    encode_from_utf8("\u{20000}", Encoding::Gb18030, &mut bytes).expect("must encode");
    let err = decode_strict(Encoding::Gb18030, &bytes[..3]).expect_err("must fail");
    assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
    // Lossy mode never panics on the same truncated prefix.
    assert_eq!(decode_lossy(Encoding::Gb18030, &bytes[..3]), "\u{FFFD}");
}

#[test]
fn gb18030_malformed_two_byte_sequence_via_public_api() {
    // 0xFF is not a valid GBK/gb18030 lead byte.
    let err = decode_strict(Encoding::Gb18030, &[0xFF]).expect_err("must fail");
    assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
    assert_eq!(decode_lossy(Encoding::Gb18030, &[0xFF]), "\u{FFFD}");
}

// --- Big5 ----------------------------------------------------------------------------------------

#[test]
fn big5_ascii_compatibility_via_public_api() {
    let decoded =
        decode_strict(Encoding::Big5, b"Hello, World! 123").expect("ASCII must decode as-is");
    assert_eq!(decoded, "Hello, World! 123");
}

#[test]
fn big5_round_trips_via_public_api() {
    let mut bytes = Vec::new();
    encode_from_utf8("中文測試", Encoding::Big5, &mut bytes).expect("must encode");
    let decoded = decode_strict(Encoding::Big5, &bytes).expect("must decode");
    assert_eq!(decoded, "中文測試");
}

#[test]
fn big5_two_code_point_output_via_public_api() {
    // Pointer 1133 decodes to U+00CA U+0304 (Ê with combining macron) -- two code points from a
    // single 2-byte sequence, since Big5's index can only hold one code point per pointer.
    let decoded = decode_strict(Encoding::Big5, &[0x88, 0x62]).expect("must decode");
    assert_eq!(decoded, "\u{00CA}\u{0304}");
}

#[test]
fn big5_truncated_lead_byte_via_public_api() {
    let err = decode_strict(Encoding::Big5, &[0xA4]).expect_err("must fail");
    assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
    assert_eq!(decode_lossy(Encoding::Big5, &[0xA4]), "\u{FFFD}");
}

// --- EUC-KR --------------------------------------------------------------------------------------

#[test]
fn euc_kr_ascii_compatibility_via_public_api() {
    let decoded =
        decode_strict(Encoding::EucKr, b"Hello, World! 123").expect("ASCII must decode as-is");
    assert_eq!(decoded, "Hello, World! 123");
}

#[test]
fn euc_kr_round_trips_via_public_api() {
    let mut bytes = Vec::new();
    encode_from_utf8("안녕하세요", Encoding::EucKr, &mut bytes).expect("must encode");
    let decoded = decode_strict(Encoding::EucKr, &bytes).expect("must decode");
    assert_eq!(decoded, "안녕하세요");
}

#[test]
fn euc_kr_unified_hangul_extension_character_via_public_api() {
    // Pointer 0 (U+AC02) is only reachable via the Unified Hangul Code / Windows-949 extension,
    // not the older, smaller KS X 1001 repertoire alone.
    let decoded = decode_strict(Encoding::EucKr, &[0x81, 0x41]).expect("must decode");
    assert_eq!(decoded, "\u{AC02}");
    let mut bytes = Vec::new();
    encode_from_utf8("\u{AC02}", Encoding::EucKr, &mut bytes).expect("must encode");
    assert_eq!(bytes, vec![0x81, 0x41]);
}

#[test]
fn euc_kr_truncated_lead_byte_via_public_api() {
    let err = decode_strict(Encoding::EucKr, &[0xC7]).expect_err("must fail");
    assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
    assert_eq!(decode_lossy(Encoding::EucKr, &[0xC7]), "\u{FFFD}");
}

// --- Cross-cutting: every single byte, lossy, never panics --------------------------------------

#[test]
fn every_byte_0_255_lossy_decoded_one_at_a_time_never_panics() {
    for &encoding in &[
        Encoding::Gbk,
        Encoding::Gb18030,
        Encoding::Big5,
        Encoding::EucKr,
    ] {
        let mut decoder = encoding.new_decoder();
        let mut out = String::new();
        let mut total_consumed = 0usize;
        for b in 0u16..=255 {
            let last = b == 255;
            total_consumed += decoder.decode_to_string_lossy(&[b as u8], &mut out, last);
        }
        assert_eq!(total_consumed, 256, "{encoding}");
    }
}

// --- Cross-cutting: label resolution feeds straight into the right decoder ----------------------

#[test]
fn label_to_decoder_pipeline_for_all_four_cjk_encodings() {
    for (label, encoding, text) in [
        ("gbk", Encoding::Gbk, "中"),
        ("GBK", Encoding::Gbk, "中"),
        ("gb2312", Encoding::Gbk, "中"),
        ("gb18030", Encoding::Gb18030, "中"),
        ("GB18030", Encoding::Gb18030, "\u{20000}"),
        ("big5", Encoding::Big5, "中"),
        ("Big5", Encoding::Big5, "中"),
        ("euc-kr", Encoding::EucKr, "한"),
        ("EUC-KR", Encoding::EucKr, "한"),
        ("windows-949", Encoding::EucKr, "한"),
    ] {
        let resolved =
            Encoding::for_label(label).unwrap_or_else(|| panic!("label {label} must resolve"));
        assert_eq!(resolved, encoding, "label={label}");
        let mut bytes = Vec::new();
        encode_from_utf8(text, resolved, &mut bytes).expect("must encode");
        let decoded = decode_strict(resolved, &bytes).expect("must decode");
        assert_eq!(decoded, text, "label={label}");
    }
}
