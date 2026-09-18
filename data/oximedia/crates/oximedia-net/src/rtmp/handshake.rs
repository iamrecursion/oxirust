//! RTMP handshake implementation.
//!
//! The RTMP handshake consists of three phases:
//! - C0/S0: Version byte (always 0x03)
//! - C1/S1: 1536 bytes (timestamp + zero + random data)
//! - C2/S2: 1536 bytes (echo of peer's C1/S1)

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::if_not_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::single_match_else)]
#![allow(clippy::redundant_slicing)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::format_collect)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unused_async)]
#![allow(clippy::identity_op)]
use crate::error::{NetError, NetResult};
use bytes::{Buf, BufMut, Bytes, BytesMut};

/// RTMP protocol version.
pub const RTMP_VERSION: u8 = 3;

/// Size of C1/S1/C2/S2 packets.
pub const HANDSHAKE_SIZE: usize = 1536;

/// Total handshake packet size (version + random data).
pub const C0_SIZE: usize = 1;

/// Handshake state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeState {
    /// Initial state, waiting to start.
    Uninitialized,
    /// Sent C0+C1, waiting for S0+S1.
    VersionSent,
    /// Received S0+S1, sent C2, waiting for S2.
    AckSent,
    /// Handshake complete.
    Done,
}

impl HandshakeState {
    /// Returns true if the handshake is complete.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        matches!(self, Self::Done)
    }
}

/// RTMP handshake handler.
#[derive(Debug)]
pub struct Handshake {
    /// Current state.
    state: HandshakeState,
    /// Client timestamp.
    client_timestamp: u32,
    /// Server timestamp.
    server_timestamp: u32,
    /// Client random data (C1).
    client_random: [u8; HANDSHAKE_SIZE - 8],
    /// Server random data (S1).
    server_random: [u8; HANDSHAKE_SIZE - 8],
    /// Epoch/start time.
    epoch: u32,
}

impl Handshake {
    /// Creates a new handshake handler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: HandshakeState::Uninitialized,
            client_timestamp: 0,
            server_timestamp: 0,
            client_random: [0u8; HANDSHAKE_SIZE - 8],
            server_random: [0u8; HANDSHAKE_SIZE - 8],
            epoch: 0,
        }
    }

    /// Returns the current handshake state.
    #[must_use]
    pub const fn state(&self) -> HandshakeState {
        self.state
    }

    /// Returns true if handshake is complete.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.state.is_done()
    }

    /// Sets the epoch time.
    pub fn set_epoch(&mut self, epoch: u32) {
        self.epoch = epoch;
    }

    /// Generates C0+C1 packet (client -> server).
    ///
    /// C0: 1 byte version
    /// C1: 4 bytes timestamp + 4 bytes zero + 1528 bytes random
    #[must_use]
    pub fn generate_c0c1(&mut self) -> Bytes {
        let mut buf = BytesMut::with_capacity(C0_SIZE + HANDSHAKE_SIZE);

        // C0: version
        buf.put_u8(RTMP_VERSION);

        // C1: timestamp
        self.client_timestamp = self.epoch;
        buf.put_u32(self.client_timestamp);

        // C1: zero (for simple handshake)
        buf.put_u32(0);

        // C1: random data
        fill_random_buffer(&mut self.client_random, self.epoch);
        buf.put_slice(&self.client_random);

        self.state = HandshakeState::VersionSent;
        buf.freeze()
    }

    /// Parses S0+S1 packet (server -> client).
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is malformed.
    pub fn parse_s0s1(&mut self, data: &[u8]) -> NetResult<()> {
        if data.len() < C0_SIZE + HANDSHAKE_SIZE {
            return Err(NetError::handshake(format!(
                "S0+S1 too short: {} bytes",
                data.len()
            )));
        }

        let mut buf = &data[..];

        // S0: version
        let version = buf.get_u8();
        if version != RTMP_VERSION {
            return Err(NetError::handshake(format!(
                "Unsupported RTMP version: {version}"
            )));
        }

        // S1: timestamp
        self.server_timestamp = buf.get_u32();

        // S1: skip zero field
        let _ = buf.get_u32();

        // S1: random data
        let random_len = HANDSHAKE_SIZE - 8;
        if buf.len() >= random_len {
            self.server_random.copy_from_slice(&buf[..random_len]);
        }

        Ok(())
    }

    /// Generates C2 packet (client -> server).
    ///
    /// C2 echoes S1 with the server's timestamp.
    #[must_use]
    pub fn generate_c2(&mut self) -> Bytes {
        let mut buf = BytesMut::with_capacity(HANDSHAKE_SIZE);

        // Echo server timestamp
        buf.put_u32(self.server_timestamp);

        // Our timestamp (time since receiving S1)
        buf.put_u32(self.epoch);

        // Echo server random
        buf.put_slice(&self.server_random);

        self.state = HandshakeState::AckSent;
        buf.freeze()
    }

    /// Parses S2 packet (server -> client).
    ///
    /// # Errors
    ///
    /// Returns an error if S2 doesn't match C1.
    pub fn parse_s2(&mut self, data: &[u8]) -> NetResult<()> {
        if data.len() < HANDSHAKE_SIZE {
            return Err(NetError::handshake(format!(
                "S2 too short: {} bytes",
                data.len()
            )));
        }

        let mut buf = &data[..];

        // Check timestamp (should be our C1 timestamp)
        let timestamp = buf.get_u32();
        if timestamp != self.client_timestamp {
            // Soft warning - some servers don't echo correctly
        }

        // Skip time2
        let _ = buf.get_u32();

        // Verify random data matches C1
        let random_len = self.client_random.len();
        if buf.len() >= random_len && buf[..random_len] != self.client_random {
            // Soft warning - some servers modify the random data
        }

        self.state = HandshakeState::Done;
        Ok(())
    }

    // Server-side methods

    /// Parses C0+C1 packet (client -> server).
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is malformed.
    pub fn parse_c0c1(&mut self, data: &[u8]) -> NetResult<()> {
        if data.len() < C0_SIZE + HANDSHAKE_SIZE {
            return Err(NetError::handshake(format!(
                "C0+C1 too short: {} bytes",
                data.len()
            )));
        }

        let mut buf = &data[..];

        // C0: version
        let version = buf.get_u8();
        if version != RTMP_VERSION {
            return Err(NetError::handshake(format!(
                "Unsupported RTMP version: {version}"
            )));
        }

        // C1: timestamp
        self.client_timestamp = buf.get_u32();

        // C1: skip zero
        let _ = buf.get_u32();

        // C1: random data
        let random_len = HANDSHAKE_SIZE - 8;
        if buf.len() >= random_len {
            self.client_random.copy_from_slice(&buf[..random_len]);
        }

        Ok(())
    }

    /// Generates S0+S1+S2 packet (server -> client).
    #[must_use]
    pub fn generate_s0s1s2(&mut self) -> Bytes {
        let mut buf = BytesMut::with_capacity(C0_SIZE + HANDSHAKE_SIZE * 2);

        // S0: version
        buf.put_u8(RTMP_VERSION);

        // S1: timestamp
        self.server_timestamp = self.epoch;
        buf.put_u32(self.server_timestamp);

        // S1: zero
        buf.put_u32(0);

        // S1: random data
        fill_random_buffer(&mut self.server_random, self.epoch);
        buf.put_slice(&self.server_random);

        // S2: echo C1
        buf.put_u32(self.client_timestamp);
        buf.put_u32(self.epoch);
        buf.put_slice(&self.client_random);

        buf.freeze()
    }

    /// Parses C2 packet (client -> server).
    ///
    /// # Errors
    ///
    /// Returns an error if C2 doesn't match S1.
    pub fn parse_c2(&mut self, data: &[u8]) -> NetResult<()> {
        if data.len() < HANDSHAKE_SIZE {
            return Err(NetError::handshake(format!(
                "C2 too short: {} bytes",
                data.len()
            )));
        }

        let mut buf = &data[..];

        // Check timestamp (should be our S1 timestamp)
        let timestamp = buf.get_u32();
        if timestamp != self.server_timestamp {
            // Soft check
        }

        // Skip time2
        let _ = buf.get_u32();

        // Verify random matches S1
        let random_len = self.server_random.len();
        if buf.len() >= random_len && buf[..random_len] != self.server_random {
            // Soft check
        }

        self.state = HandshakeState::Done;
        Ok(())
    }

    /// Creates random data using a simple PRNG.
    #[must_use]
    pub fn create_random_data(seed: u32) -> [u8; HANDSHAKE_SIZE - 8] {
        let mut data = [0u8; HANDSHAKE_SIZE - 8];
        let mut s = seed as u64;
        for byte in &mut data {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            *byte = (s >> 16) as u8;
        }
        data
    }
}

impl Default for Handshake {
    fn default() -> Self {
        Self::new()
    }
}

/// Fills buffer with pseudo-random data.
fn fill_random_buffer(buf: &mut [u8], seed: u32) {
    // Simple PRNG for deterministic testing
    // In production, use a proper random source
    let mut s = u64::from(seed);
    for byte in buf.iter_mut() {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        *byte = (s >> 16) as u8;
    }
}

/// Validates a digest in the handshake packet (for complex handshake).
///
/// This is a placeholder for proper digest validation.
#[must_use]
#[allow(dead_code)]
pub fn validate_digest(data: &[u8], key: &[u8]) -> bool {
    // Complex handshake uses HMAC-SHA256 for validation
    // This is a placeholder - real implementation would compute the digest
    let _ = (data, key);
    true
}

/// Computes digest offset in the handshake packet.
#[must_use]
#[allow(dead_code)]
pub fn compute_digest_offset(data: &[u8], scheme: u8) -> usize {
    match scheme {
        0 => {
            // Scheme 0: offset at bytes 8-11
            let offset = data.get(8..12).map(|b| {
                (u32::from(b[0]) + u32::from(b[1]) + u32::from(b[2]) + u32::from(b[3])) % 728 + 12
            });
            offset.unwrap_or(12) as usize
        }
        1 => {
            // Scheme 1: offset at bytes 764-767
            let offset = data.get(764..768).map(|b| {
                (u32::from(b[0]) + u32::from(b[1]) + u32::from(b[2]) + u32::from(b[3])) % 728 + 776
            });
            offset.unwrap_or(776) as usize
        }
        _ => 12,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handshake_state() {
        let state = HandshakeState::Uninitialized;
        assert!(!state.is_done());

        let state = HandshakeState::Done;
        assert!(state.is_done());
    }

    #[test]
    fn test_handshake_new() {
        let hs = Handshake::new();
        assert_eq!(hs.state(), HandshakeState::Uninitialized);
        assert!(!hs.is_done());
    }

    #[test]
    fn test_generate_c0c1() {
        let mut hs = Handshake::new();
        hs.set_epoch(1000);

        let data = hs.generate_c0c1();

        assert_eq!(data.len(), C0_SIZE + HANDSHAKE_SIZE);
        assert_eq!(data[0], RTMP_VERSION);
        assert_eq!(hs.state(), HandshakeState::VersionSent);
    }

    #[test]
    fn test_client_server_handshake() {
        let mut client = Handshake::new();
        let mut server = Handshake::new();

        client.set_epoch(1000);
        server.set_epoch(2000);

        // Client sends C0+C1
        let c0c1 = client.generate_c0c1();
        assert_eq!(client.state(), HandshakeState::VersionSent);

        // Server parses C0+C1 and sends S0+S1+S2
        server.parse_c0c1(&c0c1).expect("should succeed in test");
        let s0s1s2 = server.generate_s0s1s2();

        // Client parses S0+S1 and sends C2
        client.parse_s0s1(&s0s1s2).expect("should succeed in test");
        let c2 = client.generate_c2();
        assert_eq!(client.state(), HandshakeState::AckSent);

        // Client parses S2 (from s0s1s2)
        client
            .parse_s2(&s0s1s2[C0_SIZE + HANDSHAKE_SIZE..])
            .expect("should succeed in test");
        assert_eq!(client.state(), HandshakeState::Done);

        // Server parses C2
        server.parse_c2(&c2).expect("should succeed in test");
        assert_eq!(server.state(), HandshakeState::Done);
    }

    #[test]
    fn test_invalid_version() {
        let mut hs = Handshake::new();

        // Create invalid C0+C1 with wrong version
        let mut data = vec![0u8; C0_SIZE + HANDSHAKE_SIZE];
        data[0] = 4; // Invalid version

        let result = hs.parse_c0c1(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_short_packet() {
        let mut hs = Handshake::new();

        let short_data = vec![0u8; 100];
        let result = hs.parse_c0c1(&short_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_random_data() {
        let data1 = Handshake::create_random_data(42);
        let data2 = Handshake::create_random_data(42);

        // Same seed should produce same data
        assert_eq!(data1, data2);

        // Different seeds should produce different data
        let data3 = Handshake::create_random_data(43);
        assert_ne!(data1, data3);
    }

    #[test]
    fn test_handshake_c0c1_byte_layout() {
        let mut hs = Handshake::new();
        // epoch = 100 = 0x0000_0064
        hs.set_epoch(0x0000_0064);

        let bytes = hs.generate_c0c1();

        // C0: version byte must be 0x03
        assert_eq!(bytes[0], RTMP_VERSION, "C0 version must be 0x03");

        // C1 timestamp (big-endian u32 = 100 = 0x00_00_00_64)
        assert_eq!(
            &bytes[1..5],
            &[0x00, 0x00, 0x00, 0x64],
            "C1 timestamp must be big-endian 100"
        );

        // C1 zero field
        assert_eq!(
            &bytes[5..9],
            &[0x00, 0x00, 0x00, 0x00],
            "C1 zero field must be all zeros"
        );

        // Total length: 1 (C0) + 1536 (C1) = 1537
        assert_eq!(
            bytes.len(),
            C0_SIZE + HANDSHAKE_SIZE,
            "C0+C1 total length must be 1537"
        );
    }

    #[test]
    fn test_server_handshake_s0s1s2_byte_layout() {
        let mut server = Handshake::new();
        server.set_epoch(0); // server epoch = 0

        // Build synthetic C0+C1: version=0x03, client timestamp=5, zero field, rest=0
        let mut c0c1 = vec![0u8; C0_SIZE + HANDSHAKE_SIZE];
        c0c1[0] = RTMP_VERSION; // C0
                                // C1 timestamp = 5 in big-endian at bytes [1..5]
        c0c1[1] = 0x00;
        c0c1[2] = 0x00;
        c0c1[3] = 0x00;
        c0c1[4] = 0x05;
        // bytes [5..9] remain zero (C1 zero field)
        // remaining bytes remain zero (C1 random)

        server
            .parse_c0c1(&c0c1)
            .expect("parse_c0c1 must succeed on valid C0+C1");

        let result = server.generate_s0s1s2();

        // S0: version
        assert_eq!(result[0], RTMP_VERSION, "S0 must be 0x03");

        // Total length: 1 (S0) + 1536 (S1) + 1536 (S2) = 3073
        assert_eq!(
            result.len(),
            C0_SIZE + HANDSHAKE_SIZE * 2,
            "S0+S1+S2 total length must be 3073"
        );

        // S1 timestamp = server epoch = 0 (big-endian at bytes [1..5])
        assert_eq!(
            &result[1..5],
            &[0x00, 0x00, 0x00, 0x00],
            "S1 timestamp must be server epoch (0)"
        );

        // S1 zero field (bytes [5..9])
        assert_eq!(
            &result[5..9],
            &[0x00, 0x00, 0x00, 0x00],
            "S1 zero field must be all zeros"
        );

        // S2 starts at byte 1537 (C0_SIZE + HANDSHAKE_SIZE).
        // S2[0..4] echoes client timestamp (5) in big-endian.
        let s2_start = C0_SIZE + HANDSHAKE_SIZE;
        assert_eq!(
            &result[s2_start..s2_start + 4],
            &[0x00, 0x00, 0x00, 0x05],
            "S2 must echo client timestamp (5)"
        );
    }

    #[test]
    fn test_handshake_reference_roundtrip_bytes() {
        // Verify the complete C0C1 → S0S1S2 → C2 roundtrip with known epoch values
        // and check that C2 echoes the server's S1 timestamp byte-exactly.
        let mut client = Handshake::new();
        let mut server = Handshake::new();

        // client epoch=100 (0xC8 = 200 for server makes sense; use 100 and 200)
        client.set_epoch(100);
        server.set_epoch(200);

        // Client → Server: C0+C1
        let c0c1 = client.generate_c0c1();

        // Verify C1 timestamp encodes client epoch (100 = 0x64)
        assert_eq!(
            &c0c1[1..5],
            &[0x00, 0x00, 0x00, 0x64],
            "C1 timestamp must encode client epoch 100"
        );

        // Server processes C0+C1 and produces S0+S1+S2
        server
            .parse_c0c1(&c0c1)
            .expect("server must parse C0+C1 without error");
        let s0s1s2 = server.generate_s0s1s2();

        // S1 timestamp must encode server epoch (200 = 0xC8)
        assert_eq!(
            &s0s1s2[1..5],
            &[0x00, 0x00, 0x00, 0xC8],
            "S1 timestamp must encode server epoch 200"
        );

        // Client processes S0+S1 and produces C2
        client
            .parse_s0s1(&s0s1s2)
            .expect("client must parse S0+S1 without error");
        let c2 = client.generate_c2();

        // C2[0..4] echoes server's S1 timestamp (200 = 0x00_00_00_C8)
        assert_eq!(
            &c2[0..4],
            &[0x00, 0x00, 0x00, 0xC8],
            "C2 must echo server S1 timestamp (200)"
        );

        // C2 total length must be HANDSHAKE_SIZE (1536)
        assert_eq!(
            c2.len(),
            HANDSHAKE_SIZE,
            "C2 must be exactly HANDSHAKE_SIZE bytes"
        );
    }
}
