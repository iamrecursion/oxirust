//! VP9 decoder fuzzer.
//!
//! This fuzzer tests the VP9 decoder for:
//! - Superframe parsing
//! - Frame header parsing
//! - Partition parsing
//! - Probability table updates
//! - Coefficient decoding
//! - Boolean decoder
//! - Loop filter
//! - Reference frame management
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_codec::{Vp9Decoder, DecoderConfig, VideoDecoder};
use oximedia_core::CodecId;

fuzz_target!(|data: &[u8]| {
    // Create decoder with VP9 config
    let config = DecoderConfig {
        codec: CodecId::Vp9,
        extradata: None,
        threads: 1,
        low_latency: false,
    };

    let mut decoder = match Vp9Decoder::new(config) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Try to decode the fuzzer input as a VP9 bitstream
    let _ = decoder.send_packet(data, 0);

    // Try to receive frames
    // Limit iterations to prevent infinite loops
    for _ in 0..100 {
        match decoder.receive_frame() {
            Ok(Some(_frame)) => {
                // Successfully decoded a frame
            }
            Ok(None) => {
                // No frame available
                break;
            }
            Err(_) => {
                // Decoding error
                break;
            }
        }
    }

    // Try to flush
    let _ = decoder.flush();
    for _ in 0..100 {
        match decoder.receive_frame() {
            Ok(Some(_frame)) => {}
            Ok(None) | Err(_) => break,
        }
    }
});
