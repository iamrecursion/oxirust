//! Y4M (YUV4MPEG2) parser fuzzer.
//!
//! This fuzzer tests the Y4M demuxer for:
//! - `YUV4MPEG2` stream-header signature and parameter parsing
//!   (W, H, F, I, A, C tags)
//! - Chroma subsampling mode parsing (420, 422, 444, mono, ...)
//! - Frame-size computation from untrusted dimensions
//! - Per-frame `FRAME` header parsing and payload reads
//! - EOF and truncated-frame handling
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory
//! safety issues (the MP4 bounds-check standard).

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_container::demux::Y4mDemuxer;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // Header parsing happens in the constructor and must never panic.
    let mut demuxer = match Y4mDemuxer::new(Cursor::new(data)) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Exercise header accessors on whatever parsed.
    let header = demuxer.header();
    let _ = header.frame_size();
    let _ = header.fps();
    let _ = header.pixel_aspect_ratio();
    let _ = demuxer.chroma();

    // Read frames until EOF or error.
    // Limit iterations to prevent infinite loops on malformed data.
    for _ in 0..1000 {
        match demuxer.read_frame() {
            Ok(Some(_frame)) => {}
            Ok(None) | Err(_) => break,
        }
    }

    let _ = demuxer.is_eof();
    let _ = demuxer.frame_count();
});
