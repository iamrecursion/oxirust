//! Pure-Rust VP8 (lossy) key-frame decoder for WebP.
//!
//! WebP lossy still images embed a single VP8 **key frame** (RFC 6386). Because
//! a key frame is intra-only, no inter prediction or motion compensation is
//! required, which bounds the decoder to the intra path: boolean entropy
//! decoding, frame-header parsing, intra prediction, DCT/WHT inverse transforms,
//! coefficient (token) decoding, the in-loop deblocking filter, and the final
//! 4:2:0 chroma upsampling with YCbCr->RGB conversion.
//!
//! # Standards
//! This is a faithful implementation of RFC 6386 ("VP8 Data Format and
//! Decoding Guide"). Every normative probability and coefficient table in
//! [`tables`] is transcribed verbatim from the RFC. VP8 is royalty-free and
//! the RFC and libvpx (BSD-3) were used only as references for understanding.
//!
//! # Scope
//! - Implemented: VP8 key-frame decode (every intra mode, both loop filters,
//!   segmentation, multi-partition token data, per-segment dequantisation).
//! - Out of scope: inter frames / motion compensation (never present in a
//!   still WebP image).
//!
//! The single public entry point is [`decode_vp8_keyframe`].

mod bool_decoder;
mod decode;
mod frame_header;
mod loopfilter;
mod predict;
mod tables;
mod transform;

pub use decode::decode_vp8_keyframe;

#[cfg(test)]
mod integration_tests {
    use super::decode_vp8_keyframe;

    #[test]
    fn test_rejects_garbage() {
        // Not a valid VP8 payload.
        assert!(decode_vp8_keyframe(&[0u8; 4]).is_err());
    }

    #[test]
    fn test_rejects_bad_start_code() {
        // 10-byte minimal payload with a valid key-frame tag but no start code.
        let mut data = vec![0u8; 32];
        let part_size: u32 = 16;
        let tag = part_size << 5; // key frame, partition size
        data[0] = (tag & 0xFF) as u8;
        data[1] = ((tag >> 8) & 0xFF) as u8;
        data[2] = ((tag >> 16) & 0xFF) as u8;
        // start code bytes left as 0 => invalid
        assert!(decode_vp8_keyframe(&data).is_err());
    }
}
