//! RTMP (Real-Time Messaging Protocol) parser fuzzer.
//!
//! This fuzzer tests the RTMP protocol parser for:
//! - AMF0 (Action Message Format) decoding:
//!   - Number (IEEE 754 double)
//!   - Boolean
//!   - String (UTF-8)
//!   - Object (key-value pairs)
//!   - Null and Undefined
//!   - ECMA Array
//!   - Strict Array
//!   - Date
//!   - Long String
//! - Chunk header parsing:
//!   - Basic header (fmt + chunk stream ID)
//!   - Message header (timestamp, length, type, stream ID)
//!   - Header types (0, 1, 2, 3)
//!   - Extended timestamp handling
//! - Chunk stream multiplexing
//! - Message assembly from chunks
//! - Malformed AMF data handling
//! - Invalid chunk formats
//! - Buffer overflow protection
//! - Nested object depth limits
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_net::rtmp::{AmfDecoder, ChunkHeader};
use std::collections::HashMap;

fuzz_target!(|data: &[u8]| {
    // Test AMF0 decoder
    let mut decoder = AmfDecoder::new(data);

    // Try to decode values until we run out of data or hit an error
    // Limit iterations to prevent infinite loops
    let mut iteration_count = 0;
    const MAX_ITERATIONS: usize = 100;

    while decoder.has_remaining() && iteration_count < MAX_ITERATIONS {
        match decoder.decode() {
            Ok(value) => {
                // Successfully decoded a value
                // Check various value types without causing panic
                let _ = value.is_number();
                let _ = value.is_boolean();
                let _ = value.is_string();
                let _ = value.is_object();
                let _ = value.is_null();
                let _ = value.as_number();
                let _ = value.as_boolean();
                let _ = value.as_str();
            }
            Err(_) => {
                // Decoding error - stop processing
                break;
            }
        }
        iteration_count += 1;
    }

    // Test chunk header parsing
    let mut buf = data;
    let prev_headers = HashMap::new();

    // Try to parse chunk header
    match ChunkHeader::decode(&mut buf, &prev_headers) {
        Ok(header) => {
            // Successfully parsed chunk header
            // Access fields without causing panic
            let _ = header.format;
            let _ = header.chunk_stream_id;
            let _ = header.timestamp();
            let _ = header.message.message_length;
            let _ = header.message.message_type;
            let _ = header.message.message_stream_id;
        }
        Err(_) => {
            // Parsing error - this is expected for random data
        }
    }

    // Test with previous headers to exercise type 1, 2, 3 chunk headers
    if data.len() > 10 {
        let mut prev_headers_with_data = HashMap::new();

        // Create a synthetic previous header
        let mut synthetic_buf = data;
        if let Ok(synthetic_header) = ChunkHeader::decode(&mut synthetic_buf, &prev_headers) {
            let csid = synthetic_header.chunk_stream_id;
            prev_headers_with_data.insert(csid, synthetic_header);

            // Try parsing with the previous header context
            let mut buf2 = data;
            let _ = ChunkHeader::decode(&mut buf2, &prev_headers_with_data);
        }
    }

    // Test AMF with various starting positions
    for offset in [0, 1, 2, 4, 8, 16, 32] {
        if data.len() > offset {
            let mut partial_decoder = AmfDecoder::new(&data[offset..]);
            for _ in 0..10 {
                if !partial_decoder.has_remaining() {
                    break;
                }
                match partial_decoder.decode() {
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
        }
    }

    // Test chunk parsing with various buffer sizes
    for chunk_size in [1, 128, 256, 512, 1024, 4096] {
        if data.len() >= chunk_size {
            let mut chunk_buf = &data[..chunk_size];
            let _ = ChunkHeader::decode(&mut chunk_buf, &prev_headers);
        }
    }

    // Test with specific AMF marker patterns
    if data.len() > 0 {
        // Test each AMF type marker
        for marker in 0..20u8 {
            let mut test_data = vec![marker];
            test_data.extend_from_slice(data);
            let mut test_decoder = AmfDecoder::new(&test_data);
            let _ = test_decoder.decode();
        }
    }

    // Test chunk header with various format bits
    if data.len() > 0 {
        for fmt in 0..4u8 {
            let mut test_data = vec![fmt << 6 | (data[0] & 0x3F)];
            if data.len() > 1 {
                test_data.extend_from_slice(&data[1..]);
            }
            let mut test_buf = test_data.as_slice();
            let _ = ChunkHeader::decode(&mut test_buf, &prev_headers);
        }
    }
});
