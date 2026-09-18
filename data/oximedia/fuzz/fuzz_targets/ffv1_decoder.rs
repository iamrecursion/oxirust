//! FFV1 video decoder fuzzer.
//!
//! This fuzzer tests the FFV1 decoder for:
//! - Configuration record (extradata) parsing and validation
//! - Range coder / Golomb-Rice entropy decoding
//! - Slice structure parsing
//! - Median prediction and context modelling
//! - Plane and chroma subsampling handling
//! - CRC validation paths
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory
//! safety issues (the MP4 bounds-check standard).

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_codec::{Ffv1Decoder, VideoDecoder};

/// FFV1 configuration records are at least 16 bytes
/// (version, colorspace, shifts, bits, ec, slices, width, height).
const CONFIG_LEN: usize = 16;

fuzz_target!(|data: &[u8]| {
    // 1) Fuzz the configuration-record parser on the whole input.
    let _ = Ffv1Decoder::with_extradata(data);

    // 2) Structured decode: treat the first 16 bytes as extradata and the
    //    remainder as a coded frame, exercising the full decode path when
    //    the configuration record happens to validate.
    if data.len() <= CONFIG_LEN {
        return;
    }

    let (extradata, packet) = data.split_at(CONFIG_LEN);
    let mut decoder = match Ffv1Decoder::with_extradata(extradata) {
        Ok(d) => d,
        Err(_) => return,
    };

    let _ = decoder.send_packet(packet, 0);

    // Drain decoded frames.
    // Limit iterations to prevent infinite loops.
    for _ in 0..100 {
        match decoder.receive_frame() {
            Ok(Some(_frame)) => {}
            Ok(None) | Err(_) => break,
        }
    }

    // Flush and drain any remaining frames.
    let _ = decoder.flush();
    for _ in 0..100 {
        match decoder.receive_frame() {
            Ok(Some(_frame)) => {}
            Ok(None) | Err(_) => break,
        }
    }
});
