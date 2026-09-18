//! Integration tests for the three Japanese multi-byte encodings, exercised purely through the
//! public API (`Encoding`, `Decoder`, `encode_from_utf8`) rather than the crate-internal `jis`
//! module directly -- this is the end-to-end wiring the mission specifically calls out: "Shift_JIS
//! 0x80, 0xA0, lead/trail error ranges; EUC-JP SS2/SS3; ISO-2022-JP escape handling incl.
//! unexpected ESC and state-reset rules."

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

// --- Shift_JIS -------------------------------------------------------------------------------

#[test]
fn shift_jis_byte_0x80_is_u_0080_via_public_api() {
    let decoded = decode_strict(Encoding::ShiftJis, &[0x80]).expect("0x80 must decode");
    assert_eq!(decoded, "\u{0080}");
}

#[test]
fn shift_jis_byte_0xa0_is_rejected_via_public_api() {
    let err = decode_strict(Encoding::ShiftJis, &[0xA0]).expect_err("0xA0 must fail");
    assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
}

#[test]
fn shift_jis_lead_trail_error_ranges_via_public_api() {
    // 0xFD, 0xFE, 0xFF are never valid Shift_JIS lead bytes.
    for lead in [0xFDu8, 0xFE, 0xFF] {
        let err = decode_strict(Encoding::ShiftJis, &[lead]).expect_err("must fail");
        assert_eq!(
            err.kind,
            DecodeErrorKind::MalformedSequence,
            "lead=0x{lead:02X}"
        );
    }
    // 0x7F is never a valid Shift_JIS trail byte (the two valid ranges are 0x40..=0x7E and
    // 0x80..=0xFC, deliberately excluding 0x7F). It *is* an ASCII byte, though, so per spec it
    // is "restored" (reprocessed fresh) rather than consumed as part of the error -- the
    // malformed unit is just the lone lead byte (length 1).
    let err = decode_strict(Encoding::ShiftJis, &[0x81, 0x7F]).expect_err("must fail");
    assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
    assert_eq!(err.offset, 0);
    assert_eq!(err.length, 1);
    // A non-ASCII, still-invalid trail byte (0xFF, above the 0x80..=0xFC range) is instead
    // consumed together with the lead byte as a single 2-byte malformed unit.
    let err = decode_strict(Encoding::ShiftJis, &[0x81, 0xFF]).expect_err("must fail");
    assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
    assert_eq!(err.offset, 0);
    assert_eq!(err.length, 2);
}

#[test]
fn shift_jis_round_trips_via_public_api() {
    let mut bytes = Vec::new();
    encode_from_utf8("日本語", Encoding::ShiftJis, &mut bytes).expect("must encode");
    let decoded = decode_strict(Encoding::ShiftJis, &bytes).expect("must decode");
    assert_eq!(decoded, "日本語");
}

// --- EUC-JP ------------------------------------------------------------------------------------

#[test]
fn euc_jp_ss2_half_width_katakana_via_public_api() {
    // SS2 (0x8E) + 0xB1 = halfwidth ｱ (U+FF71).
    let decoded = decode_strict(Encoding::EucJp, &[0x8E, 0xB1]).expect("must decode");
    assert_eq!(decoded, "\u{FF71}");
}

#[test]
fn euc_jp_ss3_jis_x_0212_via_public_api() {
    // SS3 (0x8F) + 0xA2 0xAF = JIS X 0212 pointer 108 = U+02D8 (BREVE).
    let decoded = decode_strict(Encoding::EucJp, &[0x8F, 0xA2, 0xAF]).expect("must decode");
    assert_eq!(decoded, "\u{02D8}");
}

#[test]
fn euc_jp_ss2_ss3_truncated_at_end_of_stream() {
    let err = decode_strict(Encoding::EucJp, &[0x8E]).expect_err("must fail");
    assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
    let err = decode_strict(Encoding::EucJp, &[0x8F, 0xA2]).expect_err("must fail");
    assert_eq!(err.kind, DecodeErrorKind::TruncatedSequence);
}

#[test]
fn euc_jp_lossy_never_panics_on_dangling_ss2_ss3() {
    assert_eq!(decode_lossy(Encoding::EucJp, &[0x8E]), "\u{FFFD}");
    assert_eq!(decode_lossy(Encoding::EucJp, &[0x8F, 0xA2]), "\u{FFFD}");
}

#[test]
fn euc_jp_round_trips_via_public_api() {
    let mut bytes = Vec::new();
    encode_from_utf8("日本語\u{FF71}\u{FF72}", Encoding::EucJp, &mut bytes).expect("must encode");
    let decoded = decode_strict(Encoding::EucJp, &bytes).expect("must decode");
    assert_eq!(decoded, "日本語\u{FF71}\u{FF72}");
}

// --- ISO-2022-JP -------------------------------------------------------------------------------

#[test]
fn iso2022jp_unexpected_esc_via_public_api() {
    // ESC followed by a byte that is neither 0x24 nor 0x28.
    let err = decode_strict(Encoding::Iso2022Jp, &[0x1B, b'Z']).expect_err("must fail");
    assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
    // Lossy mode recovers and reprocesses the offending byte as plain ASCII.
    assert_eq!(
        decode_lossy(Encoding::Iso2022Jp, &[0x1B, b'Z']),
        "\u{FFFD}Z"
    );
}

#[test]
fn iso2022jp_state_resets_after_a_strict_error() {
    let mut decoder = Encoding::Iso2022Jp.new_decoder();
    let mut out = String::new();
    // Shift to jis0208, buffer a leading byte, then feed something that can never complete it.
    decoder
        .decode_to_string(&[0x1B, 0x24, 0x42, 0x46], &mut out, false)
        .expect("must decode (not yet complete)");
    assert!(decoder.has_pending_state());
    let err = decoder
        .decode_to_string(&[0x00], &mut out, true)
        .expect_err("must fail");
    assert_eq!(err.kind, DecodeErrorKind::MalformedSequence);
    // Per the `decode_to_string` contract, carried state is reset to empty after an error, so
    // the same `Decoder` is safely reusable for a fresh call afterward.
    assert!(!decoder.has_pending_state());
    let mut out2 = String::new();
    decoder
        .decode_to_string(b"OK", &mut out2, true)
        .expect("must decode after reset");
    assert_eq!(out2, "OK");
}

#[test]
fn iso2022jp_round_trips_via_public_api() {
    let mut bytes = Vec::new();
    encode_from_utf8(
        "Hello 日本語 \u{00A5}100 world",
        Encoding::Iso2022Jp,
        &mut bytes,
    )
    .expect("must encode");
    let decoded = decode_strict(Encoding::Iso2022Jp, &bytes).expect("must decode");
    assert_eq!(decoded, "Hello 日本語 \u{00A5}100 world");
}

// --- Cross-cutting: label resolution feeds straight into the right decoder --------------------

#[test]
fn label_to_decoder_pipeline_for_all_three_japanese_encodings() {
    for (label, encoding) in [
        ("shift_jis", Encoding::ShiftJis),
        ("Shift_JIS", Encoding::ShiftJis),
        ("sjis", Encoding::ShiftJis),
        ("windows-31j", Encoding::ShiftJis),
        ("euc-jp", Encoding::EucJp),
        ("EUC-JP", Encoding::EucJp),
        ("iso-2022-jp", Encoding::Iso2022Jp),
        ("ISO-2022-JP", Encoding::Iso2022Jp),
    ] {
        let resolved =
            Encoding::for_label(label).unwrap_or_else(|| panic!("label {label} must resolve"));
        assert_eq!(resolved, encoding, "label={label}");
        let mut bytes = Vec::new();
        encode_from_utf8("日", resolved, &mut bytes).expect("must encode");
        let decoded = decode_strict(resolved, &bytes).expect("must decode");
        assert_eq!(decoded, "日", "label={label}");
    }
}
