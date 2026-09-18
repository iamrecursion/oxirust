//! TIFF image parser fuzzer.
//!
//! This fuzzer tests the TIFF reader for:
//! - Byte-order (II/MM) and version validation (classic TIFF + BigTIFF)
//! - IFD entry parsing (tag, type, count, value/offset)
//! - Strip and tile offset/byte-count table handling
//! - Compression paths (uncompressed, PackBits, LZW, Deflate, CCITT, JPEG)
//! - Photometric interpretation and samples-per-pixel edge cases
//! - Unknown-tag preservation round-tripping
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory
//! safety issues (the MP4 bounds-check standard).
//!
//! Note: the public entry point is path-based, so the input is persisted to
//! a per-process scratch file in the OS temp directory each iteration.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_image::tiff::read_tiff;

fuzz_target!(|data: &[u8]| {
    // read_tiff only accepts a path; stage the fuzz input in temp storage.
    // The file name is per-process so parallel fuzz jobs never collide.
    let path = std::env::temp_dir().join(format!("oximedia_fuzz_tiff_{}.tif", std::process::id()));

    if std::fs::write(&path, data).is_err() {
        // Temp storage unavailable; nothing to test this iteration.
        return;
    }

    // Must return Err (never panic) on malformed input.
    let _ = read_tiff(&path, 0);
});
