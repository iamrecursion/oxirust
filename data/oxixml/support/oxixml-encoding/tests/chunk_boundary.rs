//! Integration tests exercised purely through the public API of `oxixml-encoding`.
//!
//! The centerpiece is the "chunk-boundary torture test": for every supported [`Encoding`], take
//! a fixture string within that encoding's repertoire, encode it, then feed the resulting bytes
//! back through an incremental [`Decoder`] one byte, two bytes, and three bytes at a time. The
//! reassembled output must equal the original string exactly, proving that no partial multi-byte
//! sequence (or, for UTF-16, surrogate pair) is ever lost or corrupted at a chunk boundary.

use oxixml_encoding::{
    ByteOrderGuess, DecodeError, Encoding, detect_bom, encode_from_utf8, sniff_xml_encoding,
};

/// Feed `bytes` through a fresh decoder for `encoding`, `chunk_size` bytes at a time, and return
/// the fully reassembled string (or the first error encountered).
fn decode_in_chunks(
    encoding: Encoding,
    bytes: &[u8],
    chunk_size: usize,
) -> Result<String, DecodeError> {
    assert!(chunk_size > 0, "chunk_size must be positive");
    let mut decoder = encoding.new_decoder();
    let mut out = String::new();
    let mut offset = 0;
    loop {
        let end = (offset + chunk_size).min(bytes.len());
        let last = end == bytes.len();
        decoder.decode_to_string(&bytes[offset..end], &mut out, last)?;
        offset = end;
        if last {
            break;
        }
    }
    Ok(out)
}

/// Round-trip `text` through `encoding` at every chunk size in `1..=3`, asserting the result
/// matches `text` exactly.
fn assert_round_trips_at_every_chunk_size(text: &str, encoding: Encoding) {
    let mut bytes = Vec::new();
    encode_from_utf8(text, encoding, &mut bytes)
        .unwrap_or_else(|e| panic!("{encoding}: {text:?} must be representable: {e}"));
    for chunk_size in [1usize, 2, 3] {
        let decoded = decode_in_chunks(encoding, &bytes, chunk_size).unwrap_or_else(|e| {
            panic!("{encoding}: decode at chunk_size={chunk_size} failed: {e}")
        });
        assert_eq!(
            decoded, text,
            "{encoding}: mismatch at chunk_size={chunk_size}"
        );
    }
    // The un-chunked, single-shot decode must also round-trip (sanity check on the fixture
    // itself, independent of chunking).
    let whole = decode_in_chunks(encoding, &bytes, bytes.len().max(1))
        .unwrap_or_else(|e| panic!("{encoding}: whole-buffer decode failed: {e}"));
    assert_eq!(whole, text, "{encoding}: whole-buffer mismatch");
}

#[test]
fn utf8_chunk_boundary_torture() {
    // ASCII, BMP (arrows, Japanese), and astral (emoji, mathematical alphanumeric) mixed.
    let fixture = "The quick brown fox 🦊 jumps over → the lazy dog. こんにちは 🎉🎊 café 𝔘𝔫𝔦𝔠𝔬𝔡𝔢";
    assert_round_trips_at_every_chunk_size(fixture, Encoding::Utf8);
}

#[test]
fn utf16_le_chunk_boundary_torture() {
    let fixture = "The quick brown fox 🦊 jumps over → the lazy dog. こんにちは 🎉🎊 café 𝔘𝔫𝔦𝔠𝔬𝔡𝔢";
    assert_round_trips_at_every_chunk_size(fixture, Encoding::Utf16Le);
}

#[test]
fn utf16_be_chunk_boundary_torture() {
    let fixture = "The quick brown fox 🦊 jumps over → the lazy dog. こんにちは 🎉🎊 café 𝔘𝔫𝔦𝔠𝔬𝔡𝔢";
    assert_round_trips_at_every_chunk_size(fixture, Encoding::Utf16Be);
}

#[test]
fn latin1_chunk_boundary_torture() {
    // Restricted to U+0000..=U+00FF, ISO-8859-1's entire repertoire.
    let fixture = "Fa\u{E7}ade na\u{EF}ve r\u{E9}sum\u{E9} caf\u{E9} \u{BD} \u{A9} \u{DF} \u{C6}";
    assert_round_trips_at_every_chunk_size(fixture, Encoding::Latin1);
}

#[test]
fn windows1252_chunk_boundary_torture() {
    // Includes the windows-1252-specific 0x80..=0x9F block: smart quotes, em dash, euro, ellipsis.
    let fixture =
        "\u{201C}Hello\u{201D} \u{2014} caf\u{E9} costs \u{20AC}3.50\u{2026} r\u{E9}sum\u{E9}";
    assert_round_trips_at_every_chunk_size(fixture, Encoding::Windows1252);
}

#[test]
fn ascii_chunk_boundary_torture() {
    let fixture = "Hello, World! 123 #$%^&*() The quick brown fox jumps over the lazy dog.";
    assert_round_trips_at_every_chunk_size(fixture, Encoding::Ascii);
}

#[test]
fn legacy_single_byte_chunk_boundary_torture() {
    // One fixture per legacy single-byte encoding, restricted to bytes/characters that
    // encoding's WHATWG index actually maps (each encoding has its own repertoire, including
    // gaps for a few of them -- see `oxixml_encoding::Encoding::ALL`).
    let cases: &[(oxixml_encoding::Encoding, &str)] = &[
        (
            oxixml_encoding::Encoding::Iso8859_2,
            "Za\u{017C}\u{00F3}\u{0142}\u{0107} g\u{0119}\u{015B}l\u{0105} ja\u{017A}\u{0144}",
        ),
        (
            oxixml_encoding::Encoding::Iso8859_5,
            "\u{0417}\u{0434}\u{0440}\u{0430}\u{0432}\u{0441}\u{0442}\u{0432}\u{0443}\u{0439}\u{0442}\u{0435}",
        ),
        (
            oxixml_encoding::Encoding::Iso8859_7,
            "\u{0393}\u{03B5}\u{03B9}\u{03AC} \u{03C3}\u{03B1}\u{03C2}",
        ),
        (
            oxixml_encoding::Encoding::Iso8859_9,
            "\u{011E}\u{0130}\u{015E} T\u{00FC}rk\u{00E7}e",
        ),
        (
            oxixml_encoding::Encoding::Windows1251,
            "\u{041F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}",
        ),
        (
            oxixml_encoding::Encoding::Koi8R,
            "\u{041F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}",
        ),
        (
            oxixml_encoding::Encoding::Macintosh,
            "Caf\u{00E9} na\u{00EF}ve",
        ),
    ];
    for &(encoding, fixture) in cases {
        assert_round_trips_at_every_chunk_size(fixture, encoding);
    }
}

#[test]
fn shift_jis_chunk_boundary_torture() {
    let fixture = "Hello, \u{4E16}\u{754C}! \u{65E5}\u{672C}\u{8A9E}\u{30C6}\u{30B9}\u{30C8} \u{FF71}\u{FF72}\u{FF73} 123";
    assert_round_trips_at_every_chunk_size(fixture, Encoding::ShiftJis);
}

#[test]
fn euc_jp_chunk_boundary_torture() {
    let fixture = "Hello, \u{4E16}\u{754C}! \u{65E5}\u{672C}\u{8A9E}\u{30C6}\u{30B9}\u{30C8} \u{FF71}\u{FF72}\u{FF73} 123";
    assert_round_trips_at_every_chunk_size(fixture, Encoding::EucJp);
}

#[test]
fn iso_2022_jp_chunk_boundary_torture() {
    let fixture = "Hello, \u{4E16}\u{754C}! \u{65E5}\u{672C}\u{8A9E}\u{30C6}\u{30B9}\u{30C8} end";
    assert_round_trips_at_every_chunk_size(fixture, Encoding::Iso2022Jp);
}

#[test]
fn gbk_chunk_boundary_torture() {
    // Restricted to GBK's 2-byte repertoire (no supplementary-plane characters).
    let fixture = "Hello, \u{4E2D}\u{6587}! \u{4F60}\u{597D}\u{4E16}\u{754C} \u{20AC} 123";
    assert_round_trips_at_every_chunk_size(fixture, Encoding::Gbk);
}

#[test]
fn gb18030_chunk_boundary_torture() {
    // Mixes ASCII, 2-byte, and 4-byte (including a supplementary-plane) code points -- gb18030
    // is total, so nothing here needs to be repertoire-restricted.
    let fixture =
        "Hello, \u{4E2D}\u{6587}! \u{4F60}\u{597D}\u{4E16}\u{754C} \u{20AC} \u{20000} 123";
    assert_round_trips_at_every_chunk_size(fixture, Encoding::Gb18030);
}

#[test]
fn big5_chunk_boundary_torture() {
    let fixture = "Hello, \u{4E2D}\u{6587}! \u{4F60}\u{597D}\u{4E16}\u{754C} 123";
    assert_round_trips_at_every_chunk_size(fixture, Encoding::Big5);
}

#[test]
fn euc_kr_chunk_boundary_torture() {
    let fixture = "Hello, \u{D55C}\u{AD6D}\u{C5B4} \u{C548}\u{B155}\u{D558}\u{C138}\u{C694} 123";
    assert_round_trips_at_every_chunk_size(fixture, Encoding::EucKr);
}

/// Decode `bytes` through `encoding` in `chunk_size`-byte pieces until an error occurs, and
/// return that error together with the number of bytes that had already been fully consumed by
/// *prior, successful* calls — i.e. the offset, within the whole `bytes` buffer, of the start of
/// the chunk that failed. `chunk_start + error.offset` is therefore the malformed byte's true
/// position in `bytes`, independent of how the input happened to be sliced into chunks (per the
/// contract documented on [`oxixml_encoding::DecodeError`]).
fn decode_in_chunks_expect_error(
    encoding: Encoding,
    bytes: &[u8],
    chunk_size: usize,
) -> (DecodeError, usize) {
    let mut decoder = encoding.new_decoder();
    let mut out = String::new();
    let mut offset = 0usize;
    loop {
        let end = (offset + chunk_size).min(bytes.len());
        let last = end == bytes.len();
        match decoder.decode_to_string(&bytes[offset..end], &mut out, last) {
            Ok(_) => {
                assert!(
                    !last,
                    "expected an error but the whole buffer decoded successfully"
                );
                offset = end;
            }
            Err(err) => return (err, offset),
        }
    }
}

#[test]
fn utf8_malformed_sequence_reports_correct_offset_regardless_of_chunking() {
    // "ok" (2 valid bytes) then a lone continuation byte 0x80 (invalid as a lead byte) at
    // global offset 2.
    let bytes: &[u8] = b"ok\x80rest";
    for chunk_size in [1usize, 2, 3, 100] {
        let (err, chunk_start) = decode_in_chunks_expect_error(Encoding::Utf8, bytes, chunk_size);
        assert_eq!(chunk_start + err.offset, 2, "chunk_size={chunk_size}");
    }
}

#[test]
fn ascii_disallowed_byte_reports_correct_offset_regardless_of_chunking() {
    // 0xFF at global offset 3.
    let bytes: &[u8] = b"abc\xFFdef";
    for chunk_size in [1usize, 2, 3, 100] {
        let (err, chunk_start) = decode_in_chunks_expect_error(Encoding::Ascii, bytes, chunk_size);
        assert_eq!(chunk_start + err.offset, 3, "chunk_size={chunk_size}");
    }
}

#[test]
fn utf16_unpaired_surrogate_is_rejected_regardless_of_chunking() {
    // High surrogate (0xD800) not followed by a low surrogate, little-endian.
    let bytes: &[u8] = &[0x00, 0xD8, b'x', 0x00];
    for chunk_size in [1usize, 2, 3, 100] {
        decode_in_chunks(Encoding::Utf16Le, bytes, chunk_size)
            .expect_err(&format!("chunk_size={chunk_size} must fail"));
    }
}

#[test]
fn label_round_trip_for_every_encoding() {
    for &encoding in Encoding::ALL {
        assert_eq!(Encoding::for_label(encoding.name()), Some(encoding));
        // Labels must resolve case-insensitively too.
        assert_eq!(
            Encoding::for_label(&encoding.name().to_lowercase()),
            Some(encoding)
        );
        assert_eq!(
            Encoding::for_label(&encoding.name().to_uppercase()),
            Some(encoding)
        );
    }
}

#[test]
fn bom_detection_matches_sniff_result_for_all_three_signatures() {
    let cases: &[(&[u8], Encoding, usize)] = &[
        (&[0xEF, 0xBB, 0xBF, b'<'], Encoding::Utf8, 3),
        (&[0xFE, 0xFF, 0x00, b'<'], Encoding::Utf16Be, 2),
        (&[0xFF, 0xFE, b'<', 0x00], Encoding::Utf16Le, 2),
    ];
    for &(bytes, expected_encoding, expected_len) in cases {
        let (encoding, len) = detect_bom(bytes).expect("BOM must be detected");
        assert_eq!(encoding, expected_encoding);
        assert_eq!(len, expected_len);

        let sniffed = sniff_xml_encoding(bytes);
        assert_eq!(sniffed.bom_len, expected_len);
    }
}

#[test]
fn sniff_appendix_f_patterns_end_to_end() {
    // 3C 3F 78 6D: "<?xm" as ASCII/UTF-8/Latin-1/windows-1252.
    let result = sniff_xml_encoding(b"<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?><r/>");
    assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Utf8));
    assert_eq!(result.best_effort_encoding(), Encoding::Latin1);

    // 00 3C 00 3F: "<?xm" as UTF-16BE without a BOM.
    let mut utf16be = Vec::new();
    for unit in "<?xml version=\"1.0\"?><r/>".encode_utf16() {
        utf16be.extend_from_slice(&unit.to_be_bytes());
    }
    let result = sniff_xml_encoding(&utf16be);
    assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Utf16Be));
    assert_eq!(result.best_effort_encoding(), Encoding::Utf16Be);

    // 3C 00 3F 00: "<?xm" as UTF-16LE without a BOM.
    let mut utf16le = Vec::new();
    for unit in "<?xml version=\"1.0\"?><r/>".encode_utf16() {
        utf16le.extend_from_slice(&unit.to_le_bytes());
    }
    let result = sniff_xml_encoding(&utf16le);
    assert_eq!(result.byte_order_guess, Some(ByteOrderGuess::Utf16Le));
    assert_eq!(result.best_effort_encoding(), Encoding::Utf16Le);
}

#[test]
fn lossy_decode_never_errors_on_arbitrary_bytes() {
    // Every possible byte value, decoded lossily through every encoding, one byte at a time:
    // must never panic and must always consume every byte.
    let all_bytes: Vec<u8> = (0..=255).collect();
    for &encoding in Encoding::ALL {
        let mut decoder = encoding.new_decoder();
        let mut out = String::new();
        let mut total_consumed = 0usize;
        for (i, byte) in all_bytes.iter().enumerate() {
            let last = i + 1 == all_bytes.len();
            total_consumed +=
                decoder.decode_to_string_lossy(std::slice::from_ref(byte), &mut out, last);
        }
        assert_eq!(total_consumed, all_bytes.len(), "{encoding}");
    }
}
