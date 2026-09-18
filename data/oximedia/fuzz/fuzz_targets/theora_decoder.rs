//! Theora decoder fuzzer.
//!
//! This fuzzer tests the Theora decoder for:
//! - Bitstream parsing (frame header, quality index)
//! - Intra frame decoding (I-frames/keyframes)
//! - Inter frame decoding (P-frames with motion compensation)
//! - DCT coefficient decoding (DC and AC)
//! - Huffman entropy decoding
//! - Motion vector parsing
//! - Block mode decision handling
//! - Reference frame management (last and golden frames)
//! - YUV420p plane reconstruction
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_codec::theora::{TheoraDecoder};
use oximedia_codec::traits::VideoDecoder;
use oximedia_core::PixelFormat;

fuzz_target!(|data: &[u8]| {
    // Test various resolution configurations
    // Include small, medium, and large resolutions to test different code paths
    let test_configs = [
        (64, 64),      // Tiny
        (176, 144),    // QCIF
        (320, 240),    // QVGA
        (640, 480),    // VGA
        (1280, 720),   // 720p
    ];

    for (width, height) in test_configs {
        let mut decoder = match TheoraDecoder::new(width, height, PixelFormat::Yuv420p) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Try to decode the fuzzer input as a Theora bitstream
        // This should never panic even on completely invalid data
        let _ = decoder.send_packet(data, 0);

        // Try to receive frames
        // Limit iterations to prevent infinite loops
        for _ in 0..50 {
            match decoder.receive_frame() {
                Ok(Some(_frame)) => {
                    // Successfully decoded a frame
                    // Check frame properties without causing panic
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
        for _ in 0..10 {
            match decoder.receive_frame() {
                Ok(Some(_frame)) => {
                    // Flushed frame
                }
                Ok(None) | Err(_) => {
                    break;
                }
            }
        }

        // Test reset functionality
        decoder.reset();
    }
});
