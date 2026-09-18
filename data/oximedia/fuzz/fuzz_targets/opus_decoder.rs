//! Opus decoder fuzzer.
//!
//! This fuzzer tests the Opus decoder for:
//! - TOC (Table of Contents) byte parsing
//! - SILK mode decoding
//! - CELT mode decoding
//! - Hybrid SILK+CELT decoding
//! - Range decoder
//! - Packet structure parsing
//! - Multiple frame configurations
//! - CBR/VBR handling
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_audio::{OpusDecoder, AudioDecoderConfig, AudioDecoder};
use oximedia_core::CodecId;

fuzz_target!(|data: &[u8]| {
    // Create decoder with Opus config
    // Test various sample rates and channel configurations
    for sample_rate in [8000, 12000, 16000, 24000, 48000] {
        for channels in [1, 2] {
            let config = AudioDecoderConfig {
                codec: CodecId::Opus,
                sample_rate,
                channels,
                extradata: None,
            };

            let mut decoder = match OpusDecoder::new(&config) {
                Ok(d) => d,
                Err(_) => continue,
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
        }
    }
});
