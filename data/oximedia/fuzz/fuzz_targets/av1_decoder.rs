//! AV1 decoder fuzzer.
//!
//! This fuzzer tests the AV1 decoder for:
//! - OBU (Open Bitstream Unit) parsing
//! - Sequence header parsing
//! - Frame header parsing
//! - Tile decoding
//! - Transform coefficient decoding
//! - Entropy decoding (symbol reader)
//! - Loop filter and CDEF
//! - Reference frame management
//! - Film grain synthesis
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_codec::{Av1Decoder, DecoderConfig, VideoDecoder};

fuzz_target!(|data: &[u8]| {
    // Create decoder with default config
    let config = DecoderConfig::default();
    let mut decoder = match Av1Decoder::new(config) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Try to decode the fuzzer input as an AV1 bitstream
    // This should never panic even on completely invalid data
    let _ = decoder.send_packet(data, 0);

    // Try to receive frames
    // Limit iterations to prevent infinite loops
    for _ in 0..100 {
        match decoder.receive_frame() {
            Ok(Some(_frame)) => {
                // Successfully decoded a frame
            }
            Ok(None) => {
                // No frame available, need more data
                break;
            }
            Err(_) => {
                // Decoding error - stop processing
                break;
            }
        }
    }

    // Try to flush any remaining frames
    let _ = decoder.flush();
    for _ in 0..100 {
        match decoder.receive_frame() {
            Ok(Some(_frame)) => {
                // Flushed frame
            }
            Ok(None) | Err(_) => {
                break;
            }
        }
    }
});
