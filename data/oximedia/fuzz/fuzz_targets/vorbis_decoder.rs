//! Vorbis decoder fuzzer.
//!
//! This fuzzer tests the Vorbis decoder for:
//! - Identification header parsing
//! - Comment header parsing
//! - Setup header parsing (codebooks, floors, residues, mappings)
//! - Audio packet decoding
//! - MDCT transform
//! - Floor curve decoding
//! - Residue decoding
//! - Channel coupling
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_audio::{VorbisDecoder, AudioDecoderConfig, AudioDecoder};
use oximedia_core::CodecId;

fuzz_target!(|data: &[u8]| {
    // Vorbis requires extradata (headers), but for fuzzing we'll test
    // with various configurations including missing headers
    for has_extradata in [false, true] {
        let config = AudioDecoderConfig {
            codec: CodecId::Vorbis,
            sample_rate: 44100,
            channels: 2,
            extradata: if has_extradata {
                // Use fuzzer data as potential extradata
                Some(data.to_vec())
            } else {
                None
            },
        };

        let mut decoder = match VorbisDecoder::new(&config) {
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
});
