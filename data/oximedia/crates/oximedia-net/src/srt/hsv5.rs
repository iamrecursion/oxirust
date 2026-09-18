//! SRT HSv5 handshake extensions and enriched configuration types.
//!
//! This module adds:
//! - [`SrtEncryption`] — encryption algorithm selection
//! - [`SrtPacketType`] — high-level packet type classification
//! - [`SrtHandshake`] — HSv5 handshake structure with SRT magic (0x4A17)
//! - [`SrtStreamConfig`] — richer session configuration
//! - [`SrtExtensionBlock`] — SRT handshake extension block encoding/decoding

#![allow(dead_code)]

use super::packet::ControlType;

// ── SrtEncryption ─────────────────────────────────────────────────────────────

/// Encryption algorithm for SRT streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SrtEncryption {
    /// No encryption.
    #[default]
    None,
    /// AES-128 (128-bit key).
    Aes128,
    /// AES-256 (256-bit key).
    Aes256,
}

impl SrtEncryption {
    /// Returns the key length in bytes for this algorithm.
    ///
    /// Returns `0` for [`SrtEncryption::None`].
    #[must_use]
    pub const fn key_len_bytes(self) -> usize {
        match self {
            Self::None => 0,
            Self::Aes128 => 16,
            Self::Aes256 => 32,
        }
    }

    /// Returns `true` if encryption is enabled.
    #[must_use]
    pub const fn is_encrypted(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Returns the SRT encryption field value (used in handshake).
    ///
    /// - `0` = no encryption
    /// - `2` = AES-128
    /// - `3` = AES-256
    #[must_use]
    pub const fn handshake_field(self) -> u16 {
        match self {
            Self::None => 0,
            Self::Aes128 => 2,
            Self::Aes256 => 3,
        }
    }

    /// Reconstruct from the handshake field value.
    #[must_use]
    pub const fn from_handshake_field(v: u16) -> Self {
        match v {
            2 => Self::Aes128,
            3 => Self::Aes256,
            _ => Self::None,
        }
    }
}

// ── SrtPacketType ─────────────────────────────────────────────────────────────

/// High-level SRT packet type classification.
///
/// Mirrors the distinct packet kinds visible to protocol-level consumers,
/// combining the data/control distinction with common control subtypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtPacketType {
    /// A payload data packet.
    Data,
    /// A generic control packet (catch-all).
    Control,
    /// A keepalive control packet.
    Keepalive,
    /// An acknowledgement (ACK) control packet.
    Ack,
    /// A negative acknowledgement (NAK) control packet.
    Nak,
    /// A shutdown control packet.
    Shutdown,
    /// A handshake control packet.
    Handshake,
    /// An ACK-ACK control packet.
    AckAck,
    /// A drop request control packet.
    DropReq,
}

impl SrtPacketType {
    /// Derive a [`SrtPacketType`] from a [`ControlType`].
    #[must_use]
    pub fn from_control_type(ct: &ControlType) -> Self {
        match ct {
            ControlType::Keepalive => Self::Keepalive,
            ControlType::Ack => Self::Ack,
            ControlType::Nak => Self::Nak,
            ControlType::Shutdown => Self::Shutdown,
            ControlType::Handshake => Self::Handshake,
            ControlType::AckAck => Self::AckAck,
            ControlType::DropReq => Self::DropReq,
            _ => Self::Control,
        }
    }

    /// Returns `true` if this is a data packet.
    #[must_use]
    pub const fn is_data(self) -> bool {
        matches!(self, Self::Data)
    }

    /// Returns `true` if this is any kind of control packet.
    #[must_use]
    pub const fn is_control(self) -> bool {
        !matches!(self, Self::Data)
    }

    /// Returns `true` if this packet type carries a handshake payload.
    #[must_use]
    pub const fn is_handshake(self) -> bool {
        matches!(self, Self::Handshake)
    }
}

// ── SrtHandshake ──────────────────────────────────────────────────────────────

/// SRT HSv5 handshake structure.
///
/// Encodes the fields defined in the SRT protocol specification for HSv5,
/// including the SRT magic number (`0x4A17`) in the version field prefix and
/// the standard UDT-derived fields.
///
/// Wire layout (all big-endian, 48 bytes):
/// ```text
///  0                   1                   2                   3
///  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                          Magic (4A17)                         |  [0..4]
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |           Version             |         Extension Field       |  [4..8]
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                      Initial Seq Number                       |  [8..12]
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                      Max Packet Size (MTU)                    |  [12..16]
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                        Flow Window Size                       |  [16..20]
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                       Handshake Type                          |  [20..24]
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                         Socket ID                             |  [24..28]
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                          SYN Cookie                           |  [28..32]
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                   Peer IPv4 Address (or 0)                    |  [32..36]
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                       Reserved (12 bytes)                     |  [36..48]
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtHandshake {
    /// SRT magic number. Must equal [`SrtHandshake::SRT_MAGIC`] (`0x4A17`).
    pub magic: u32,
    /// SRT protocol version. Format: `0x00MMNNPP` (major, minor, patch).
    /// Default is `0x00010300` (SRT 1.3.0).
    pub version: u32,
    /// Extension field / flags bitmask (low 16 bits used for SRT extensions).
    pub extension_field: u16,
    /// Initial packet sequence number.
    pub initial_seq: u32,
    /// Maximum transmission unit size in bytes (default 1500).
    pub mtu: u32,
    /// Maximum flow window size (default 8192 packets).
    pub flow_window: u32,
    /// Handshake type (waveahand=0, induction=1, conclusion=-1, agreement=-2).
    pub handshake_type: i32,
    /// Local SRT socket ID.
    pub socket_id: u32,
    /// SYN cookie for replay protection.
    pub syn_cookie: u32,
    /// Peer IPv4 address in host byte order, or `0`.
    pub peer_ip_v4: u32,
}

impl SrtHandshake {
    /// SRT magic number embedded in the handshake.
    pub const SRT_MAGIC: u32 = 0x4A17;

    /// SRT version 1.3.0 encoded as `0x00010300`.
    pub const VERSION_1_3_0: u32 = 0x0001_0300;

    /// Handshake type: caller waves hand (initial).
    pub const TYPE_WAVEAHAND: i32 = 0;
    /// Handshake type: listener induction response.
    pub const TYPE_INDUCTION: i32 = 1;
    /// Handshake type: conclusion (agreement negotiation).
    pub const TYPE_CONCLUSION: i32 = -1;
    /// Handshake type: agreement (established).
    pub const TYPE_AGREEMENT: i32 = -2;

    /// Wire encoding size in bytes.
    pub const WIRE_SIZE: usize = 48;

    /// Create a new SRT HSv5 handshake with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            magic: Self::SRT_MAGIC,
            version: Self::VERSION_1_3_0,
            extension_field: 0,
            initial_seq: 0,
            mtu: 1500,
            flow_window: 8192,
            handshake_type: Self::TYPE_WAVEAHAND,
            socket_id: 0,
            syn_cookie: 0,
            peer_ip_v4: 0,
        }
    }

    /// Builder: set the socket ID.
    #[must_use]
    pub const fn with_socket_id(mut self, id: u32) -> Self {
        self.socket_id = id;
        self
    }

    /// Builder: set the SYN cookie.
    #[must_use]
    pub const fn with_syn_cookie(mut self, cookie: u32) -> Self {
        self.syn_cookie = cookie;
        self
    }

    /// Builder: set the handshake type.
    #[must_use]
    pub const fn with_handshake_type(mut self, ht: i32) -> Self {
        self.handshake_type = ht;
        self
    }

    /// Builder: encode a target latency (ms) into the upper 16 bits of
    /// [`extension_field`][Self::extension_field].
    ///
    /// Latency values > 65535 ms are clamped to 65535.
    #[must_use]
    pub fn with_latency_extension(mut self, latency_ms: u32) -> Self {
        let clamped = latency_ms.min(u32::from(u16::MAX)) as u16;
        self.extension_field = (self.extension_field & 0x00FF) | (clamped << 8);
        self
    }

    /// Returns `true` if the magic field equals [`SrtHandshake::SRT_MAGIC`].
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.magic == Self::SRT_MAGIC
    }

    /// Serialize this handshake to a 48-byte buffer.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::WIRE_SIZE);

        buf.extend_from_slice(&self.magic.to_be_bytes());
        // Pack version (16-bit) + extension_field (16-bit) into 4 bytes
        let version_lo = (self.version & 0xFFFF) as u16;
        buf.extend_from_slice(&version_lo.to_be_bytes());
        buf.extend_from_slice(&self.extension_field.to_be_bytes());
        buf.extend_from_slice(&self.initial_seq.to_be_bytes());
        buf.extend_from_slice(&self.mtu.to_be_bytes());
        buf.extend_from_slice(&self.flow_window.to_be_bytes());
        buf.extend_from_slice(&self.handshake_type.to_be_bytes());
        buf.extend_from_slice(&self.socket_id.to_be_bytes());
        buf.extend_from_slice(&self.syn_cookie.to_be_bytes());
        buf.extend_from_slice(&self.peer_ip_v4.to_be_bytes());
        // Pad to WIRE_SIZE (48 bytes)
        while buf.len() < Self::WIRE_SIZE {
            buf.push(0u8);
        }
        buf
    }

    /// Deserialize an `SrtHandshake` from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the buffer is shorter than [`SrtHandshake::WIRE_SIZE`]
    /// or if the magic number does not match [`SrtHandshake::SRT_MAGIC`].
    pub fn decode(data: &[u8]) -> Result<Self, String> {
        if data.len() < Self::WIRE_SIZE {
            return Err(format!(
                "SrtHandshake::decode: buffer too short ({} < {})",
                data.len(),
                Self::WIRE_SIZE
            ));
        }

        let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        if magic != Self::SRT_MAGIC {
            return Err(format!(
                "SrtHandshake::decode: invalid magic 0x{magic:04X} (expected 0x{:04X})",
                Self::SRT_MAGIC
            ));
        }

        let version_lo = u16::from_be_bytes([data[4], data[5]]);
        let extension_field = u16::from_be_bytes([data[6], data[7]]);
        let initial_seq = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        let mtu = u32::from_be_bytes([data[12], data[13], data[14], data[15]]);
        let flow_window = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let handshake_type = i32::from_be_bytes([data[20], data[21], data[22], data[23]]);
        let socket_id = u32::from_be_bytes([data[24], data[25], data[26], data[27]]);
        let syn_cookie = u32::from_be_bytes([data[28], data[29], data[30], data[31]]);
        let peer_ip_v4 = u32::from_be_bytes([data[32], data[33], data[34], data[35]]);

        Ok(Self {
            magic,
            version: u32::from(version_lo),
            extension_field,
            initial_seq,
            mtu,
            flow_window,
            handshake_type,
            socket_id,
            syn_cookie,
            peer_ip_v4,
        })
    }
}

impl Default for SrtHandshake {
    fn default() -> Self {
        Self::new()
    }
}

// ── SrtStreamConfig ───────────────────────────────────────────────────────────

/// Rich per-stream configuration for an SRT session.
///
/// Complements the existing [`super::socket::SrtConfig`] with higher-level
/// options such as bandwidth cap, encryption, stream ID, and payload sizing.
#[derive(Debug, Clone)]
pub struct SrtStreamConfig {
    /// Target receive latency in milliseconds (default 120).
    pub latency_ms: u32,
    /// Maximum bandwidth in bits-per-second. `0` = unlimited.
    pub max_bandwidth: u64,
    /// Encryption algorithm to use.
    pub encryption: SrtEncryption,
    /// Passphrase for key derivation (required when encryption is enabled).
    pub passphrase: Option<String>,
    /// Optional stream identifier for multiplexing (SRT SID extension).
    pub stream_id: Option<String>,
    /// Maximum SRT payload size in bytes (default 1316 = 7 × 188 TS packets).
    pub max_payload_size: u16,
}

impl Default for SrtStreamConfig {
    fn default() -> Self {
        Self {
            latency_ms: 120,
            max_bandwidth: 0,
            encryption: SrtEncryption::None,
            passphrase: None,
            stream_id: None,
            max_payload_size: 1316,
        }
    }
}

impl SrtStreamConfig {
    /// Create a new config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set the target latency.
    #[must_use]
    pub const fn with_latency(mut self, ms: u32) -> Self {
        self.latency_ms = ms;
        self
    }

    /// Builder: set the maximum bandwidth (bps).
    #[must_use]
    pub const fn with_max_bandwidth(mut self, bps: u64) -> Self {
        self.max_bandwidth = bps;
        self
    }

    /// Builder: set the encryption algorithm.
    #[must_use]
    pub const fn with_encryption(mut self, enc: SrtEncryption) -> Self {
        self.encryption = enc;
        self
    }

    /// Builder: set the passphrase.
    #[must_use]
    pub fn with_passphrase(mut self, passphrase: &str) -> Self {
        self.passphrase = Some(passphrase.to_owned());
        self
    }

    /// Builder: set the stream identifier.
    #[must_use]
    pub fn with_stream_id(mut self, id: &str) -> Self {
        self.stream_id = Some(id.to_owned());
        self
    }

    /// Builder: set the maximum payload size.
    #[must_use]
    pub const fn with_max_payload_size(mut self, size: u16) -> Self {
        self.max_payload_size = size;
        self
    }

    /// Returns `true` if encryption is configured.
    #[must_use]
    pub const fn is_encrypted(&self) -> bool {
        self.encryption.is_encrypted()
    }

    /// Returns `true` if a stream ID is set.
    #[must_use]
    pub fn has_stream_id(&self) -> bool {
        self.stream_id.is_some()
    }
}

// ── SrtExtensionBlock ─────────────────────────────────────────────────────────

/// SRT handshake extension block.
///
/// Extension blocks are appended after the base handshake and carry
/// SRT-specific metadata such as SRT version, latency, crypto keys, and
/// stream IDs.
///
/// Wire format:
/// ```text
///  0               1               2               3
///  0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |          ext_type (16)        |          ext_len  (16)        |
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |                    data  (ext_len × 4 bytes)                  |
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtExtensionBlock {
    /// Extension type identifier.
    pub ext_type: u16,
    /// Extension length in 32-bit words (not counting the 4-byte header).
    pub ext_len: u16,
    /// Raw extension data (`ext_len × 4` bytes).
    pub data: Vec<u8>,
}

impl SrtExtensionBlock {
    /// Extension type: SRT Handshake Request.
    pub const EXT_HSREQ: u16 = 1;
    /// Extension type: SRT Handshake Response.
    pub const EXT_HSRSP: u16 = 2;
    /// Extension type: Key Material Request.
    pub const EXT_KMREQ: u16 = 3;
    /// Extension type: Key Material Response.
    pub const EXT_KMRSP: u16 = 4;
    /// Extension type: Stream ID.
    pub const EXT_SID: u16 = 5;

    /// Create a new extension block, automatically computing `ext_len`.
    ///
    /// `data` is zero-padded to the next 4-byte boundary.
    #[must_use]
    pub fn new(ext_type: u16, data: Vec<u8>) -> Self {
        let word_count = (data.len() + 3) / 4;
        let ext_len = word_count as u16;
        Self {
            ext_type,
            ext_len,
            data,
        }
    }

    /// Create an SRT Handshake Request (HSREQ) block with SRT version and flags.
    ///
    /// Layout (12 bytes = 3 words):
    /// - `[0..4]`: SRT version (`0x00010300`)
    /// - `[4..8]`: SRT flags
    /// - `[8..12]`: Latency (ms) packed as `(recv_latency << 16) | send_latency`
    #[must_use]
    pub fn hsreq(srt_version: u32, srt_flags: u32, latency_ms: u16) -> Self {
        let mut data = Vec::with_capacity(12);
        data.extend_from_slice(&srt_version.to_be_bytes());
        data.extend_from_slice(&srt_flags.to_be_bytes());
        let latency_field = (u32::from(latency_ms) << 16) | u32::from(latency_ms);
        data.extend_from_slice(&latency_field.to_be_bytes());
        Self::new(Self::EXT_HSREQ, data)
    }

    /// Create a Stream ID extension block from a UTF-8 string.
    #[must_use]
    pub fn stream_id(sid: &str) -> Self {
        Self::new(Self::EXT_SID, sid.as_bytes().to_vec())
    }

    /// Serialize this extension block to bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        // Pad data to ext_len * 4 bytes
        let padded_len = usize::from(self.ext_len) * 4;
        let mut buf = Vec::with_capacity(4 + padded_len);
        buf.extend_from_slice(&self.ext_type.to_be_bytes());
        buf.extend_from_slice(&self.ext_len.to_be_bytes());
        buf.extend_from_slice(&self.data);
        // Zero-pad to full word boundary
        while buf.len() < 4 + padded_len {
            buf.push(0u8);
        }
        buf
    }

    /// Deserialize an `SrtExtensionBlock` from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the buffer is too short for the declared length.
    pub fn decode(data: &[u8]) -> Result<Self, String> {
        if data.len() < 4 {
            return Err(format!(
                "SrtExtensionBlock::decode: buffer too short ({} < 4)",
                data.len()
            ));
        }

        let ext_type = u16::from_be_bytes([data[0], data[1]]);
        let ext_len = u16::from_be_bytes([data[2], data[3]]);
        let payload_len = usize::from(ext_len) * 4;

        if data.len() < 4 + payload_len {
            return Err(format!(
                "SrtExtensionBlock::decode: payload truncated (need {} got {})",
                4 + payload_len,
                data.len()
            ));
        }

        let payload = data[4..4 + payload_len].to_vec();
        Ok(Self {
            ext_type,
            ext_len,
            data: payload,
        })
    }

    /// Returns the total wire size in bytes (4-byte header + padded payload).
    #[must_use]
    pub fn wire_size(&self) -> usize {
        4 + usize::from(self.ext_len) * 4
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SrtEncryption ──────────────────────────────────────────────────────

    #[test]
    fn test_encryption_key_len() {
        assert_eq!(SrtEncryption::None.key_len_bytes(), 0);
        assert_eq!(SrtEncryption::Aes128.key_len_bytes(), 16);
        assert_eq!(SrtEncryption::Aes256.key_len_bytes(), 32);
    }

    #[test]
    fn test_encryption_is_encrypted() {
        assert!(!SrtEncryption::None.is_encrypted());
        assert!(SrtEncryption::Aes128.is_encrypted());
        assert!(SrtEncryption::Aes256.is_encrypted());
    }

    #[test]
    fn test_encryption_handshake_field_roundtrip() {
        for enc in [
            SrtEncryption::None,
            SrtEncryption::Aes128,
            SrtEncryption::Aes256,
        ] {
            let field = enc.handshake_field();
            let decoded = SrtEncryption::from_handshake_field(field);
            assert_eq!(decoded, enc);
        }
    }

    #[test]
    fn test_encryption_default_is_none() {
        assert_eq!(SrtEncryption::default(), SrtEncryption::None);
    }

    // ── SrtPacketType ──────────────────────────────────────────────────────

    #[test]
    fn test_packet_type_data_flags() {
        assert!(SrtPacketType::Data.is_data());
        assert!(!SrtPacketType::Data.is_control());
        assert!(!SrtPacketType::Data.is_handshake());
    }

    #[test]
    fn test_packet_type_control_flags() {
        for pt in [
            SrtPacketType::Control,
            SrtPacketType::Keepalive,
            SrtPacketType::Ack,
            SrtPacketType::Nak,
            SrtPacketType::Shutdown,
            SrtPacketType::Handshake,
            SrtPacketType::AckAck,
            SrtPacketType::DropReq,
        ] {
            assert!(!pt.is_data(), "{pt:?} should not be data");
            assert!(pt.is_control(), "{pt:?} should be control");
        }
    }

    #[test]
    fn test_packet_type_handshake_flag() {
        assert!(SrtPacketType::Handshake.is_handshake());
        assert!(!SrtPacketType::Keepalive.is_handshake());
    }

    #[test]
    fn test_packet_type_from_control_type() {
        assert_eq!(
            SrtPacketType::from_control_type(&ControlType::Handshake),
            SrtPacketType::Handshake
        );
        assert_eq!(
            SrtPacketType::from_control_type(&ControlType::Keepalive),
            SrtPacketType::Keepalive
        );
        assert_eq!(
            SrtPacketType::from_control_type(&ControlType::Ack),
            SrtPacketType::Ack
        );
        assert_eq!(
            SrtPacketType::from_control_type(&ControlType::Nak),
            SrtPacketType::Nak
        );
        assert_eq!(
            SrtPacketType::from_control_type(&ControlType::Shutdown),
            SrtPacketType::Shutdown
        );
        assert_eq!(
            SrtPacketType::from_control_type(&ControlType::AckAck),
            SrtPacketType::AckAck
        );
        assert_eq!(
            SrtPacketType::from_control_type(&ControlType::DropReq),
            SrtPacketType::DropReq
        );
        // Catch-all
        assert_eq!(
            SrtPacketType::from_control_type(&ControlType::UserDefined),
            SrtPacketType::Control
        );
    }

    // ── SrtHandshake ───────────────────────────────────────────────────────

    #[test]
    fn test_handshake_new_defaults() {
        let hs = SrtHandshake::new();
        assert_eq!(hs.magic, SrtHandshake::SRT_MAGIC);
        assert_eq!(hs.version, SrtHandshake::VERSION_1_3_0);
        assert_eq!(hs.mtu, 1500);
        assert_eq!(hs.flow_window, 8192);
        assert!(hs.is_valid());
    }

    #[test]
    fn test_handshake_encode_decode_roundtrip() {
        let hs = SrtHandshake::new()
            .with_socket_id(42)
            .with_syn_cookie(0xDEAD_BEEF)
            .with_handshake_type(SrtHandshake::TYPE_INDUCTION);

        let encoded = hs.encode();
        assert_eq!(encoded.len(), SrtHandshake::WIRE_SIZE);

        let decoded = SrtHandshake::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.magic, SrtHandshake::SRT_MAGIC);
        assert_eq!(decoded.socket_id, 42);
        assert_eq!(decoded.syn_cookie, 0xDEAD_BEEF);
        assert_eq!(decoded.handshake_type, SrtHandshake::TYPE_INDUCTION);
        assert!(decoded.is_valid());
    }

    #[test]
    fn test_handshake_decode_too_short() {
        let err = SrtHandshake::decode(&[0u8; 10]);
        assert!(err.is_err());
    }

    #[test]
    fn test_handshake_decode_wrong_magic() {
        let mut buf = SrtHandshake::new().encode();
        // Corrupt the magic bytes
        buf[0] = 0xFF;
        buf[1] = 0xFF;
        buf[2] = 0xFF;
        buf[3] = 0xFF;
        let err = SrtHandshake::decode(&buf);
        assert!(err.is_err());
    }

    #[test]
    fn test_handshake_with_latency_extension() {
        let hs = SrtHandshake::new().with_latency_extension(200);
        // Upper 8 bits of extension_field should encode 200
        let latency = (hs.extension_field >> 8) as u32;
        assert_eq!(latency, 200);
    }

    #[test]
    fn test_handshake_is_valid_wrong_magic() {
        let hs = SrtHandshake {
            magic: 0x0000,
            ..SrtHandshake::new()
        };
        assert!(!hs.is_valid());
    }

    // ── SrtStreamConfig ────────────────────────────────────────────────────

    #[test]
    fn test_stream_config_defaults() {
        let cfg = SrtStreamConfig::new();
        assert_eq!(cfg.latency_ms, 120);
        assert_eq!(cfg.max_bandwidth, 0);
        assert_eq!(cfg.encryption, SrtEncryption::None);
        assert!(!cfg.is_encrypted());
        assert!(!cfg.has_stream_id());
        assert_eq!(cfg.max_payload_size, 1316);
    }

    #[test]
    fn test_stream_config_builder_chain() {
        let cfg = SrtStreamConfig::new()
            .with_latency(500)
            .with_max_bandwidth(10_000_000)
            .with_encryption(SrtEncryption::Aes256)
            .with_passphrase("s3cr3t")
            .with_stream_id("live/stream1")
            .with_max_payload_size(188);

        assert_eq!(cfg.latency_ms, 500);
        assert_eq!(cfg.max_bandwidth, 10_000_000);
        assert_eq!(cfg.encryption, SrtEncryption::Aes256);
        assert!(cfg.is_encrypted());
        assert_eq!(cfg.passphrase.as_deref(), Some("s3cr3t"));
        assert_eq!(cfg.stream_id.as_deref(), Some("live/stream1"));
        assert!(cfg.has_stream_id());
        assert_eq!(cfg.max_payload_size, 188);
    }

    // ── SrtExtensionBlock ──────────────────────────────────────────────────

    #[test]
    fn test_extension_block_new_computes_ext_len() {
        let block = SrtExtensionBlock::new(SrtExtensionBlock::EXT_HSREQ, vec![0u8; 12]);
        assert_eq!(block.ext_len, 3); // 12 bytes = 3 words
        assert_eq!(block.wire_size(), 16); // 4 header + 12 data
    }

    #[test]
    fn test_extension_block_encode_decode_roundtrip() {
        let original = SrtExtensionBlock::new(SrtExtensionBlock::EXT_SID, b"live/test".to_vec());
        let encoded = original.encode();
        let decoded = SrtExtensionBlock::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.ext_type, SrtExtensionBlock::EXT_SID);
        assert_eq!(decoded.ext_len, original.ext_len);
    }

    #[test]
    fn test_extension_block_hsreq_factory() {
        let block = SrtExtensionBlock::hsreq(SrtHandshake::VERSION_1_3_0, 0x00BF, 120);
        assert_eq!(block.ext_type, SrtExtensionBlock::EXT_HSREQ);
        assert_eq!(block.ext_len, 3); // 12 bytes = 3 words
    }

    #[test]
    fn test_extension_block_stream_id_factory() {
        let block = SrtExtensionBlock::stream_id("media/channel1");
        assert_eq!(block.ext_type, SrtExtensionBlock::EXT_SID);
        assert!(!block.data.is_empty());
    }

    #[test]
    fn test_extension_block_decode_too_short() {
        let err = SrtExtensionBlock::decode(&[0u8; 2]);
        assert!(err.is_err());
    }

    #[test]
    fn test_extension_block_decode_truncated_payload() {
        // ext_len=4 (16 bytes) but only 8 bytes provided
        let buf = [
            0x00, 0x05, // ext_type = EXT_SID (5)
            0x00, 0x04, // ext_len = 4 words = 16 bytes
            0x61, 0x62, 0x63, 0x64, // only 4 bytes of payload
        ];
        let err = SrtExtensionBlock::decode(&buf);
        assert!(err.is_err());
    }

    #[test]
    fn test_extension_type_constants() {
        assert_eq!(SrtExtensionBlock::EXT_HSREQ, 1);
        assert_eq!(SrtExtensionBlock::EXT_HSRSP, 2);
        assert_eq!(SrtExtensionBlock::EXT_KMREQ, 3);
        assert_eq!(SrtExtensionBlock::EXT_KMRSP, 4);
        assert_eq!(SrtExtensionBlock::EXT_SID, 5);
    }
}
