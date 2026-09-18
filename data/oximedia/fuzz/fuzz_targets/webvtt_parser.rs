//! WebVTT subtitle parser fuzzer.
//!
//! This fuzzer tests the WebVTT parser for:
//! - `WEBVTT` signature and header-line handling (incl. BOM)
//! - Cue timing parsing (`HH:MM:SS.mmm --> HH:MM:SS.mmm`) and settings
//! - NOTE / STYLE / REGION block handling
//! - Cue identifier lines and payload accumulation
//! - Full document parsing including regions and styles
//! - Lossy/invalid UTF-8 handling on the bytes entry point
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory
//! safety issues (the MP4 bounds-check standard).

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_subtitle::parser::webvtt;

fuzz_target!(|data: &[u8]| {
    // Bytes entry point.
    let _ = webvtt::parse(data);

    // String entry points when the input happens to be valid UTF-8.
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = webvtt::parse_webvtt(text);
        let _ = webvtt::parse_document(text);
    }
});
