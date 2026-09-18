//! AAF (Advanced Authoring Format) parser fuzzer.
//!
//! This fuzzer tests the AAF reader for:
//! - CFB/structured-storage container parsing (sector tables, FAT,
//!   directory entries)
//! - Header, dictionary, and content-storage object model traversal
//! - Property value decoding (local sets)
//! - Mob and slot graph reading
//! - Essence data extraction
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory
//! safety issues (the MP4 bounds-check standard).

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_aaf::AafReader;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // Structured-storage parsing happens in the constructor.
    let mut reader = match AafReader::new(Cursor::new(data)) {
        Ok(r) => r,
        Err(_) => return,
    };

    // Full file read: header, dictionary, content storage, essence.
    // Must return Err (never panic) on malformed input.
    let _ = reader.read();
});
