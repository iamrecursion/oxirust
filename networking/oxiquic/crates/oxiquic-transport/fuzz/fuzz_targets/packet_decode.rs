#![no_main]
//! Fuzzes [`oxiquic_transport::packet::parse_long_packet`]: header-protection
//! removal, packet-number recovery and AEAD decryption of a long-header
//! (Initial/Handshake/0-RTT) packet.
//!
//! Initial packets are the interesting case for an attacker: their keys are
//! derived purely from the (attacker-chosen) destination connection ID via a
//! public HKDF salt (RFC 9001 Section 5.2), not from any handshake secret, so
//! *anyone* can construct a datagram this function will attempt to fully
//! decode -- this is the exact code path a server runs on every unsolicited
//! Initial packet before it knows anything about the sender. This target
//! mirrors that: it derives Initial keys from a fuzzer-chosen DCID and feeds
//! the rest of the input straight to the parser.
//!
//! The target must never panic, and must never read or write outside the
//! `datagram` buffer it is given. Rejecting the input with `Err(PacketError)`
//! is the expected, safe outcome for almost all inputs.

use libfuzzer_sys::fuzz_target;
use oxiquic_crypto::quic::AES128_GCM;
use oxiquic_crypto::suites::tls13_aes_128_gcm_sha256_internal;
use oxiquic_transport::packet;
use rustls::quic::{Keys, Version};
use rustls::Side;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    // First byte picks a DCID length in 1..=20 (RFC 9000 Section 17.2); the
    // next `dcid_len` bytes (or fewer, if the input is short) are the DCID
    // used to derive Initial keys, exactly as a server derives keys from the
    // client-chosen DCID on the first Initial packet it ever sees for a
    // connection. Everything remaining is the datagram to decode.
    let dcid_len = 1 + (data[0] as usize % 20);
    let rest = &data[1..];
    let split = dcid_len.min(rest.len());
    let (dcid, datagram) = rest.split_at(split);
    if dcid.is_empty() || datagram.is_empty() {
        return;
    }

    // Both directions are worth fuzzing: a server decrypting a client
    // Initial (`Side::Server`, `remote` keys) and a client decrypting a
    // server Initial (`Side::Client`). Side only affects the traffic-label
    // used in the internal HKDF-Expand-Label, so cover both cheaply.
    for side in [Side::Server, Side::Client] {
        let keys = Keys::initial(
            Version::V1,
            tls13_aes_128_gcm_sha256_internal(),
            &AES128_GCM,
            dcid,
            side,
        );
        let mut buf = datagram.to_vec();
        let _ = packet::parse_long_packet(
            &mut buf,
            0,
            1, // QUICv1
            None,
            keys.remote.packet.as_ref(),
            keys.remote.header.as_ref(),
        );
    }
});
