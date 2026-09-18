#![no_main]
//! Fuzzes [`oxiquic_transport::packet::peek_dcid`], the very first thing an
//! `OxiQUIC` endpoint does with a freshly-received, completely unauthenticated
//! UDP datagram: classify the packet type and pull out the destination
//! connection ID so the demux loop can route it to a connection (or, for a
//! server, decide whether to attempt a new handshake). This runs before any
//! decryption or key lookup, so every byte here is fully attacker-controlled.
//!
//! The target must never panic. Any input, however malformed or truncated,
//! should come back as `Ok` or `Err(PacketError)` -- both are fine; a panic
//! or an out-of-bounds read is the bug this target exists to catch.

use libfuzzer_sys::fuzz_target;
use oxiquic_transport::packet;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    // The real caller (the server/client demux loop) knows its own local
    // short-header DCID length ahead of time; drive it from one input byte so
    // the fuzzer explores mismatches between the advertised length and what
    // is actually present, not just a single fixed length.
    let (short_dcid_len, datagram) = data.split_at(1);
    let short_dcid_len = (short_dcid_len[0] % 21) as usize; // RFC 9000: CIDs are <= 20 bytes.
    let _ = packet::peek_dcid(datagram, short_dcid_len);
});
