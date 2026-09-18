//! EBML parser fuzzer.
//!
//! This fuzzer tests the EBML (Extensible Binary Meta Language) parser for:
//! - VINT (Variable-length Integer) parsing
//! - Element ID parsing (1-4 bytes)
//! - Element size parsing with unknown size handling
//! - Element header parsing
//! - Unsigned integer reading (uint)
//! - Signed integer reading (int)
//! - String parsing (UTF-8)
//! - Float parsing (4 and 8 byte floats)
//! - Date parsing
//! - Binary data reading
//! - Nested element structures
//! - Malformed VINT encoding
//! - Integer overflow handling
//! - Incomplete data handling
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_container::demux::matroska::ebml;

fuzz_target!(|data: &[u8]| {
    // Test VINT parsing
    let _ = ebml::parse_vint(data);

    // Test element ID parsing
    let _ = ebml::parse_element_id(data);

    // Test element size parsing
    let _ = ebml::parse_element_size(data);

    // Test full element header parsing
    let _ = ebml::parse_element_header(data);

    // Test integer parsing
    let _ = ebml::read_uint(data);
    let _ = ebml::read_int(data);

    // Test string parsing with various lengths
    for len in [0, 1, 10, 100, 1000, 10000] {
        if data.len() >= len {
            let _ = ebml::read_string(data, len);
        }
    }

    // Test float parsing with 4 and 8 byte lengths
    if data.len() >= 4 {
        let _ = ebml::read_float(data, 4);
    }
    if data.len() >= 8 {
        let _ = ebml::read_float(data, 8);
    }

    // Test date parsing
    if data.len() >= 8 {
        let _ = ebml::read_date(data, 8);
    }

    // Test binary data reading with various lengths
    for len in [0, 1, 10, 100, 1000] {
        if data.len() >= len {
            let _ = ebml::read_binary(data, len);
        }
    }

    // Test element name lookup (should never panic on any u32 value)
    if data.len() >= 4 {
        let id = u32::from_be_bytes([
            data[0],
            data.get(1).copied().unwrap_or(0),
            data.get(2).copied().unwrap_or(0),
            data.get(3).copied().unwrap_or(0),
        ]);
        let _ = ebml::element_name(id);
    }

    // Test parsing chains - parsing element after element
    let mut remaining = data;
    let mut iteration_count = 0;
    const MAX_ITERATIONS: usize = 1000;

    while !remaining.is_empty() && iteration_count < MAX_ITERATIONS {
        match ebml::parse_element_header(remaining) {
            Ok((rest, element)) => {
                // Successfully parsed element header
                // Try to consume the element data
                let size = element.size.min(rest.len() as u64);
                if size > 0 && rest.len() >= size as usize {
                    remaining = &rest[size as usize..];
                } else {
                    remaining = rest;
                }

                // If no progress, break to prevent infinite loop
                if remaining.len() == rest.len() {
                    break;
                }
            }
            Err(_) => {
                // Parsing error - stop
                break;
            }
        }
        iteration_count += 1;
    }

    // Test nested structures by attempting to parse elements recursively
    if let Ok((_, element)) = ebml::parse_element_header(data) {
        let size = element.size.min(data.len() as u64);
        if size > 0 && data.len() >= size as usize {
            let element_data = &data[..size as usize];

            // Try to parse the element payload as containing more elements
            let _ = ebml::parse_element_header(element_data);

            // Test data type parsing on element payload
            let _ = ebml::read_uint(element_data);
            let _ = ebml::read_int(element_data);
            let _ = ebml::read_string(element_data, element_data.len().min(1000));
        }
    }
});
