//! FLAC audio decoder fuzzer.
//!
//! This fuzzer tests the FLAC audio decoder for:
//! - Stream marker detection
//! - Metadata block parsing
//! - Frame header parsing
//! - Subframe types (CONSTANT, VERBATIM, FIXED, LPC)
//! - Rice/Golomb coding
//! - CRC validation
//! - Channel decorrelation
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_audio::{FlacDecoder, AudioDecoderConfig, AudioDecoder};
use oximedia_core::CodecId;

fuzz_target!(|data: &[u8]| {
    // FLAC can have various sample rates and bit depths
    let config = AudioDecoderConfig {
        codec: CodecId::Flac,
        sample_rate: 44100,
        channels: 2,
        extradata: None,
    };

    let mut decoder = match FlacDecoder::new(&config) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Try to decode the fuzzer input
    let _ = decoder.send_packet(data, 0);

    // Try to receive frames
    // Limit iterations to prevent infinite loops
    for _ in 0..10 {
        match decoder.receive_frame() {
            Ok(Some(_frame)) => {
                // Successfully decoded
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
});
