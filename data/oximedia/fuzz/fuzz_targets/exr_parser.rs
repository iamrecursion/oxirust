//! OpenEXR image parser fuzzer.
//!
//! This fuzzer tests the EXR readers for:
//! - Magic number and version/flags validation
//! - Attribute header parsing (channels, dataWindow, compression, ...)
//! - Scanline offset table handling
//! - Tiled image data paths
//! - Multi-part EXR document parsing (`MultiPartExr::from_bytes`)
//! - Compression decode paths (none, RLE, ZIP, PIZ, ...)
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory
//! safety issues (the MP4 bounds-check standard).

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_image::exr::read_exr;
use oximedia_image::exr_multipart::MultiPartExr;

fuzz_target!(|data: &[u8]| {
    // ── Bytes-based multi-part parser (no I/O) ──────────────────────────
    let _ = MultiPartExr::is_exr(data);
    let _ = MultiPartExr::is_multipart_bytes(data);
    if let Ok(doc) = MultiPartExr::from_bytes(data) {
        // Exercise post-parse validation on whatever survived parsing.
        let _ = doc.validate();
        let _ = doc.part_count();
    }

    // ── Path-based single-part reader (scanline + tiled paths) ─────────
    // read_exr only accepts a path; stage the input in temp storage.
    // The file name is per-process so parallel fuzz jobs never collide.
    let path = std::env::temp_dir().join(format!("oximedia_fuzz_exr_{}.exr", std::process::id()));

    if std::fs::write(&path, data).is_err() {
        // Temp storage unavailable; the bytes-based half still ran.
        return;
    }

    // Must return Err (never panic) on malformed input.
    let _ = read_exr(&path, 0);
});
