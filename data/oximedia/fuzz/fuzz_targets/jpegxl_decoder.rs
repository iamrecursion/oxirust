//! JPEG-XL decoder fuzzer.
//!
//! This fuzzer tests the JPEG-XL (ISO/IEC 18181) decoder for:
//! - Codestream (0xFF 0x0A) and ISOBMFF container signature detection
//! - Size header and image metadata parsing
//! - Modular mode entropy decoding (ANS)
//! - Animation header parsing and multi-frame decode
//! - Streaming ISOBMFF box iteration (`JxlStreamingDecoder`)
//!
//! The fuzzer should never panic, enter infinite loops, or cause memory
//! safety issues (the MP4 bounds-check standard).

#![no_main]

use libfuzzer_sys::fuzz_target;
use oximedia_codec::jpegxl::{JxlDecoder, JxlStreamingDecoder};
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    let decoder = JxlDecoder::new();

    // Signature probes must be total functions over arbitrary bytes.
    let _ = JxlDecoder::is_jxl(data);
    let _ = JxlDecoder::is_codestream(data);
    let _ = JxlDecoder::is_container(data);

    // Header-only parse (cheap path, exercised even when decode bails).
    let _ = decoder.read_header(data);

    // Full still-image decode.
    let _ = decoder.decode(data);

    // Animation paths.
    let _ = decoder.is_animated(data);
    let _ = decoder.read_animation_header(data);
    let _ = decoder.decode_animated(data);

    // Streaming decoder over the same input (ISOBMFF auto-detection).
    // Limit iterations to prevent infinite loops on malformed data.
    if let Ok(streaming) = JxlStreamingDecoder::new(Cursor::new(data)) {
        for frame in streaming.take(100) {
            if frame.is_err() {
                break;
            }
        }
    }
});
