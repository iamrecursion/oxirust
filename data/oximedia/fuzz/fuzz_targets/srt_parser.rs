//! SubRip (SRT) subtitle parser fuzzer.
//!
//! This fuzzer tests the SRT parser for:
//! - Cue index / timestamp line parsing (`HH:MM:SS,mmm --> HH:MM:SS,mmm`)
//! - Multi-line cue text accumulation and blank-line block splitting
//! - Inline formatting tag parsing (<b>, <i>, <u>, <font>)
//! - Lossy/invalid UTF-8 handling on the bytes entry point
//! - CRLF vs LF normalization
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory
//! safety issues (the MP4 bounds-check standard).

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_subtitle::parser::srt;

fuzz_target!(|data: &[u8]| {
    // Bytes entry points (encoding handling included).
    let _ = srt::parse(data);
    let _ = srt::parse_with_formatting(data);

    // String entry points when the input happens to be valid UTF-8.
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = srt::parse_srt(text);
        let _ = srt::parse_srt_with_formatting(text);
        let _ = srt::parse_formatted_text(text);
    }
});
