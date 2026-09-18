//! ASS/SSA subtitle parser fuzzer.
//!
//! This fuzzer tests the Advanced SubStation Alpha parser for:
//! - INI-style section parsing ([Script Info], [V4+ Styles], [Events])
//! - Format-line driven field mapping for styles and dialogue events
//! - Timestamp parsing (`H:MM:SS.cc`)
//! - Override tag block parsing (`{\b1\i1\pos(10,20)}` ...)
//! - Lossy/invalid UTF-8 handling on the bytes entry point
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory
//! safety issues (the MP4 bounds-check standard).

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_subtitle::parser::ssa;

fuzz_target!(|data: &[u8]| {
    // Bytes entry point (lossy UTF-8 conversion included).
    let _ = ssa::parse(data);

    // String entry points when the input happens to be valid UTF-8.
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = ssa::parse_ass(text);
        let _ = ssa::parse_override_tags(text);
    }
});
