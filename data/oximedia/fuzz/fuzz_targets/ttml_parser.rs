//! TTML/IMSC subtitle parser fuzzer.
//!
//! This fuzzer tests the TTML (Timed Text Markup Language) parsers for:
//! - XML document structure parsing (head/body/div/p/span)
//! - Clock-time and offset-time expression parsing
//! - Region and style attribute resolution
//! - TTML2/IMSC1 enhanced parsing (`ttml_v2::TtmlParser`)
//! - Entity and malformed-markup handling
//! - Lossy/invalid UTF-8 handling on the bytes entry point
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory
//! safety issues (the MP4 bounds-check standard).

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_subtitle::parser::ttml;
use oximedia_subtitle::ttml_v2::TtmlParser;

fuzz_target!(|data: &[u8]| {
    // Bytes entry points (TTML v1 parser).
    let _ = ttml::parse(data);
    let _ = ttml::parse_ttml(data);

    // TTML2/IMSC1 parser when the input happens to be valid UTF-8.
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = TtmlParser::parse_v2(text);
        let _ = TtmlParser::parse_document(text);
    }
});
