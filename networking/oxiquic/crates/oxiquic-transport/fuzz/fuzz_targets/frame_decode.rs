#![no_main]
//! Fuzzes [`oxiquic_transport::frame::decode_frame`] over an already
//! "decrypted" payload buffer -- i.e. the trust boundary immediately inside
//! packet protection, where every one of RFC 9000's ~20 frame types (ACK
//! range lists, STREAM offsets/lengths, CRYPTO offsets, NEW_CONNECTION_ID,
//! RESET_STREAM final sizes, ...) is parsed from attacker-controlled bytes.
//!
//! A real packet payload is a back-to-back sequence of frames, so this
//! target decodes in a loop exactly like the connection's receive path does,
//! stopping at the first error (or once the buffer is exhausted).
//!
//! The target must never panic on any byte sequence, and must never read
//! past the end of the input buffer.

use libfuzzer_sys::fuzz_target;
use oxiquic_transport::coding::Buf;
use oxiquic_transport::frame;

fuzz_target!(|data: &[u8]| {
    let mut buf = Buf::new(data);
    // Cap the iteration count defensively: every successful `decode_frame`
    // call is required to advance `buf` (a zero-length PADDING coalesces
    // instead of looping), but bounding this loop keeps the fuzz target's
    // own logic obviously non-diverging even if that invariant were ever
    // violated by a future change.
    let mut iterations = 0usize;
    while buf.remaining() > 0 && iterations < data.len() + 1 {
        iterations += 1;
        if frame::decode_frame(&mut buf).is_err() {
            break;
        }
    }
});
