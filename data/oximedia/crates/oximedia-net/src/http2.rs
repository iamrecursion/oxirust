//! HTTP/2 basic frame encoding and decoding (RFC 7540).
//!
//! Implements the HTTP/2 binary framing layer: frame types, the 9-byte frame
//! header, frame encode/decode, and the SETTINGS frame payload.

use crate::error::{NetError, NetResult};

// ─── SETTINGS parameter identifiers (RFC 7540 §6.5.2) ────────────────────────

/// SETTINGS_HEADER_TABLE_SIZE (0x1) — initial value of the HPACK header table size.
pub const SETTINGS_HEADER_TABLE_SIZE: u16 = 0x1;
/// SETTINGS_ENABLE_PUSH (0x2) — whether server push is permitted.
pub const SETTINGS_ENABLE_PUSH: u16 = 0x2;
/// SETTINGS_MAX_CONCURRENT_STREAMS (0x3) — maximum number of concurrent streams.
pub const SETTINGS_MAX_CONCURRENT_STREAMS: u16 = 0x3;
/// SETTINGS_INITIAL_WINDOW_SIZE (0x4) — initial flow-control window size.
pub const SETTINGS_INITIAL_WINDOW_SIZE: u16 = 0x4;
/// SETTINGS_MAX_FRAME_SIZE (0x5) — maximum frame payload size.
pub const SETTINGS_MAX_FRAME_SIZE: u16 = 0x5;
/// SETTINGS_MAX_HEADER_LIST_SIZE (0x6) — advisory maximum size of the header list.
pub const SETTINGS_MAX_HEADER_LIST_SIZE: u16 = 0x6;

// ─── HTTP/2 connection preface ────────────────────────────────────────────────

/// The client connection preface (PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n).
pub const CLIENT_PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// Size of the HTTP/2 frame header in bytes (RFC 7540 §4.1).
pub const FRAME_HEADER_SIZE: usize = 9;

/// Maximum valid payload length (2^24 - 1 bytes).
pub const MAX_FRAME_SIZE: u32 = (1 << 24) - 1;

// ─── Frame type ───────────────────────────────────────────────────────────────

/// HTTP/2 frame type identifier (RFC 7540 §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Http2FrameType {
    /// DATA frame (0x0) — carries HTTP message body.
    Data,
    /// HEADERS frame (0x1) — opens/continues a stream, carries header block.
    Headers,
    /// PRIORITY frame (0x2) — specifies stream dependency and weight.
    Priority,
    /// RST_STREAM frame (0x3) — terminates a stream abnormally.
    RstStream,
    /// SETTINGS frame (0x4) — conveys configuration parameters.
    Settings,
    /// PUSH_PROMISE frame (0x5) — notifies the peer of an intended push.
    PushPromise,
    /// PING frame (0x6) — measures round-trip time or tests liveness.
    Ping,
    /// GOAWAY frame (0x7) — informs the peer that this endpoint is done.
    GoAway,
    /// WINDOW_UPDATE frame (0x8) — implements flow control.
    WindowUpdate,
    /// CONTINUATION frame (0x9) — continues a header block.
    Continuation,
    /// Unknown/extension frame type.
    Unknown(u8),
}

impl Http2FrameType {
    /// Convert raw byte to frame type.
    pub fn from_u8(v: u8) -> Self {
        match v {
            0x0 => Self::Data,
            0x1 => Self::Headers,
            0x2 => Self::Priority,
            0x3 => Self::RstStream,
            0x4 => Self::Settings,
            0x5 => Self::PushPromise,
            0x6 => Self::Ping,
            0x7 => Self::GoAway,
            0x8 => Self::WindowUpdate,
            0x9 => Self::Continuation,
            other => Self::Unknown(other),
        }
    }

    /// Convert frame type to raw byte.
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Data => 0x0,
            Self::Headers => 0x1,
            Self::Priority => 0x2,
            Self::RstStream => 0x3,
            Self::Settings => 0x4,
            Self::PushPromise => 0x5,
            Self::Ping => 0x6,
            Self::GoAway => 0x7,
            Self::WindowUpdate => 0x8,
            Self::Continuation => 0x9,
            Self::Unknown(v) => v,
        }
    }
}

// ─── Frame flags (RFC 7540 §6) ────────────────────────────────────────────────

/// END_STREAM flag (0x1) — applies to DATA and HEADERS.
pub const FLAG_END_STREAM: u8 = 0x1;
/// END_HEADERS flag (0x4) — applies to HEADERS, PUSH_PROMISE, CONTINUATION.
pub const FLAG_END_HEADERS: u8 = 0x4;
/// PADDED flag (0x8) — applies to DATA, HEADERS, PUSH_PROMISE.
pub const FLAG_PADDED: u8 = 0x8;
/// PRIORITY flag (0x20) — applies to HEADERS.
pub const FLAG_PRIORITY: u8 = 0x20;
/// ACK flag (0x1) — applies to SETTINGS and PING.
pub const FLAG_ACK: u8 = 0x1;

// ─── Http2Frame ───────────────────────────────────────────────────────────────

/// An HTTP/2 frame (RFC 7540 §4.1).
///
/// ```text
/// +-----------------------------------------------+
/// |                 Length (24)                    |
/// +---------------+---------------+---------------+
/// |   Type (8)    |   Flags (8)   |
/// +-+-------------+---------------+-------------------------------+
/// |R|                 Stream Identifier (31)                      |
/// +=+=============================================================+
/// |                   Frame Payload (0...)                      ...
/// +---------------------------------------------------------------+
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Http2Frame {
    /// Payload length in bytes (24-bit value, max 16 777 215).
    pub length: u32,
    /// Frame type identifier.
    pub frame_type: Http2FrameType,
    /// Type-specific flags byte.
    pub flags: u8,
    /// Stream identifier (31 bits; the reserved MSB is always cleared on encode).
    pub stream_id: u32,
    /// Frame payload bytes.
    pub payload: Vec<u8>,
}

impl Http2Frame {
    /// Construct a new frame, deriving `length` from the payload.
    pub fn new(
        frame_type: Http2FrameType,
        flags: u8,
        stream_id: u32,
        payload: Vec<u8>,
    ) -> NetResult<Self> {
        let length = payload.len() as u64;
        if length > MAX_FRAME_SIZE as u64 {
            return Err(NetError::protocol(format!(
                "Frame payload length {length} exceeds maximum {MAX_FRAME_SIZE}"
            )));
        }
        Ok(Self {
            length: length as u32,
            frame_type,
            flags,
            stream_id,
            payload,
        })
    }

    /// Encode this frame to a byte vector.
    ///
    /// The 9-byte header is followed by the payload.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(FRAME_HEADER_SIZE + self.payload.len());

        // 3-byte length (big-endian)
        buf.push((self.length >> 16) as u8);
        buf.push((self.length >> 8) as u8);
        buf.push(self.length as u8);

        // Frame type
        buf.push(self.frame_type.as_u8());

        // Flags
        buf.push(self.flags);

        // Stream identifier — reserved MSB (R) cleared, 31-bit value
        let stream_id = self.stream_id & 0x7FFF_FFFF;
        buf.push((stream_id >> 24) as u8);
        buf.push((stream_id >> 16) as u8);
        buf.push((stream_id >> 8) as u8);
        buf.push(stream_id as u8);

        // Payload
        buf.extend_from_slice(&self.payload);

        buf
    }

    /// Decode a frame from a byte slice.
    ///
    /// Returns `(frame, bytes_consumed)`. Errors if the buffer is malformed;
    /// returns `Err` with offset information when the buffer is truncated to
    /// fewer than `FRAME_HEADER_SIZE` bytes.
    pub fn decode(bytes: &[u8]) -> NetResult<(Self, usize)> {
        if bytes.len() < FRAME_HEADER_SIZE {
            return Err(NetError::parse(
                0,
                format!(
                    "Buffer too short for HTTP/2 frame header: need {FRAME_HEADER_SIZE}, got {}",
                    bytes.len()
                ),
            ));
        }

        // 3-byte payload length
        let length = (u32::from(bytes[0]) << 16) | (u32::from(bytes[1]) << 8) | u32::from(bytes[2]);

        let frame_type = Http2FrameType::from_u8(bytes[3]);
        let flags = bytes[4];

        // 4-byte stream ID with reserved MSB cleared
        let stream_id = u32::from_be_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) & 0x7FFF_FFFF;

        let payload_end = FRAME_HEADER_SIZE + length as usize;
        if bytes.len() < payload_end {
            return Err(NetError::parse(
                FRAME_HEADER_SIZE as u64,
                format!(
                    "Buffer too short for frame payload: need {payload_end}, got {}",
                    bytes.len()
                ),
            ));
        }

        let payload = bytes[FRAME_HEADER_SIZE..payload_end].to_vec();

        Ok((
            Self {
                length,
                frame_type,
                flags,
                stream_id,
                payload,
            },
            payload_end,
        ))
    }

    /// Returns `true` if the END_STREAM flag is set.
    pub fn has_end_stream(&self) -> bool {
        self.flags & FLAG_END_STREAM != 0
    }

    /// Returns `true` if the END_HEADERS flag is set.
    pub fn has_end_headers(&self) -> bool {
        self.flags & FLAG_END_HEADERS != 0
    }

    /// Returns `true` if the ACK flag is set.
    pub fn has_ack(&self) -> bool {
        self.flags & FLAG_ACK != 0
    }
}

// ─── Http2Settings ────────────────────────────────────────────────────────────

/// HTTP/2 SETTINGS parameters (RFC 7540 §6.5).
///
/// Each field is `Option<u32>` — `None` means the setting was not present in
/// the SETTINGS frame payload (peer uses the protocol default).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Http2Settings {
    /// SETTINGS_HEADER_TABLE_SIZE (default 4096).
    pub header_table_size: Option<u32>,
    /// SETTINGS_ENABLE_PUSH (default 1 = enabled).
    pub enable_push: Option<u32>,
    /// SETTINGS_MAX_CONCURRENT_STREAMS (default unlimited).
    pub max_concurrent_streams: Option<u32>,
    /// SETTINGS_INITIAL_WINDOW_SIZE (default 65 535).
    pub initial_window_size: Option<u32>,
    /// SETTINGS_MAX_FRAME_SIZE (default 16 384; valid range 2^14..2^24-1).
    pub max_frame_size: Option<u32>,
    /// SETTINGS_MAX_HEADER_LIST_SIZE (default unlimited).
    pub max_header_list_size: Option<u32>,
}

impl Http2Settings {
    /// Decode SETTINGS payload (sequence of 6-byte parameter/value pairs).
    pub fn decode(payload: &[u8]) -> NetResult<Self> {
        if payload.len() % 6 != 0 {
            return Err(NetError::parse(
                0,
                format!(
                    "SETTINGS payload length must be a multiple of 6, got {}",
                    payload.len()
                ),
            ));
        }

        let mut settings = Http2Settings::default();

        for chunk in payload.chunks(6) {
            let id = u16::from_be_bytes([chunk[0], chunk[1]]);
            let value = u32::from_be_bytes([chunk[2], chunk[3], chunk[4], chunk[5]]);

            match id {
                SETTINGS_HEADER_TABLE_SIZE => settings.header_table_size = Some(value),
                SETTINGS_ENABLE_PUSH => settings.enable_push = Some(value),
                SETTINGS_MAX_CONCURRENT_STREAMS => settings.max_concurrent_streams = Some(value),
                SETTINGS_INITIAL_WINDOW_SIZE => settings.initial_window_size = Some(value),
                SETTINGS_MAX_FRAME_SIZE => settings.max_frame_size = Some(value),
                SETTINGS_MAX_HEADER_LIST_SIZE => settings.max_header_list_size = Some(value),
                // Unknown settings MUST be ignored (RFC 7540 §6.5)
                _ => {}
            }
        }

        Ok(settings)
    }

    /// Encode only the `Some` settings into a SETTINGS payload.
    pub fn encode(&self) -> Vec<u8> {
        let entries: &[(u16, Option<u32>)] = &[
            (SETTINGS_HEADER_TABLE_SIZE, self.header_table_size),
            (SETTINGS_ENABLE_PUSH, self.enable_push),
            (SETTINGS_MAX_CONCURRENT_STREAMS, self.max_concurrent_streams),
            (SETTINGS_INITIAL_WINDOW_SIZE, self.initial_window_size),
            (SETTINGS_MAX_FRAME_SIZE, self.max_frame_size),
            (SETTINGS_MAX_HEADER_LIST_SIZE, self.max_header_list_size),
        ];

        let mut buf = Vec::new();
        for &(id, value_opt) in entries {
            if let Some(value) = value_opt {
                buf.extend_from_slice(&id.to_be_bytes());
                buf.extend_from_slice(&value.to_be_bytes());
            }
        }
        buf
    }

    /// Build a SETTINGS frame from this settings struct.
    pub fn to_frame(&self) -> NetResult<Http2Frame> {
        let payload = self.encode();
        Http2Frame::new(Http2FrameType::Settings, 0, 0, payload)
    }

    /// Build a SETTINGS ACK frame.
    pub fn ack_frame() -> NetResult<Http2Frame> {
        Http2Frame::new(Http2FrameType::Settings, FLAG_ACK, 0, vec![])
    }
}

// ─── Helper builders for common frames ───────────────────────────────────────

/// Build a PING frame (8-byte opaque payload).
pub fn ping_frame(payload: [u8; 8], ack: bool) -> NetResult<Http2Frame> {
    let flags = if ack { FLAG_ACK } else { 0 };
    Http2Frame::new(Http2FrameType::Ping, flags, 0, payload.to_vec())
}

/// Build a WINDOW_UPDATE frame for the given stream (0 = connection-level).
pub fn window_update_frame(stream_id: u32, increment: u32) -> NetResult<Http2Frame> {
    if increment == 0 || increment > 0x7FFF_FFFF {
        return Err(NetError::protocol(format!(
            "WINDOW_UPDATE increment must be 1..=2147483647, got {increment}"
        )));
    }
    let payload = (increment & 0x7FFF_FFFF).to_be_bytes().to_vec();
    Http2Frame::new(Http2FrameType::WindowUpdate, 0, stream_id, payload)
}

/// Build a RST_STREAM frame with an error code.
pub fn rst_stream_frame(stream_id: u32, error_code: u32) -> NetResult<Http2Frame> {
    Http2Frame::new(
        Http2FrameType::RstStream,
        0,
        stream_id,
        error_code.to_be_bytes().to_vec(),
    )
}

/// Build a GOAWAY frame.
pub fn goaway_frame(
    last_stream_id: u32,
    error_code: u32,
    debug_data: &[u8],
) -> NetResult<Http2Frame> {
    let mut payload = Vec::with_capacity(8 + debug_data.len());
    payload.extend_from_slice(&(last_stream_id & 0x7FFF_FFFF).to_be_bytes());
    payload.extend_from_slice(&error_code.to_be_bytes());
    payload.extend_from_slice(debug_data);
    Http2Frame::new(Http2FrameType::GoAway, 0, 0, payload)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── FrameType ────────────────────────────────────────────────────────────

    #[test]
    fn test_frame_type_roundtrip_all_known() {
        let types = [
            Http2FrameType::Data,
            Http2FrameType::Headers,
            Http2FrameType::Priority,
            Http2FrameType::RstStream,
            Http2FrameType::Settings,
            Http2FrameType::PushPromise,
            Http2FrameType::Ping,
            Http2FrameType::GoAway,
            Http2FrameType::WindowUpdate,
            Http2FrameType::Continuation,
        ];
        for t in types {
            assert_eq!(Http2FrameType::from_u8(t.as_u8()), t, "roundtrip for {t:?}");
        }
    }

    #[test]
    fn test_frame_type_unknown() {
        let t = Http2FrameType::from_u8(0xFF);
        assert!(matches!(t, Http2FrameType::Unknown(0xFF)));
        assert_eq!(t.as_u8(), 0xFF);
    }

    // ── Frame encode ─────────────────────────────────────────────────────────

    #[test]
    fn test_encode_data_frame_header() {
        let frame = Http2Frame::new(Http2FrameType::Data, FLAG_END_STREAM, 1, b"hello".to_vec())
            .expect("valid");
        let enc = frame.encode();

        // Length = 5
        assert_eq!(enc[0], 0x00);
        assert_eq!(enc[1], 0x00);
        assert_eq!(enc[2], 0x05);
        // Type = Data (0x0)
        assert_eq!(enc[3], 0x0);
        // Flags = END_STREAM
        assert_eq!(enc[4], FLAG_END_STREAM);
        // Stream ID = 1
        assert_eq!(&enc[5..9], &[0, 0, 0, 1]);
        // Payload
        assert_eq!(&enc[9..], b"hello");
    }

    #[test]
    fn test_encode_empty_payload() {
        let frame = Http2Frame::new(
            Http2FrameType::Ping,
            FLAG_ACK,
            0,
            b"\0\0\0\0\0\0\0\0".to_vec(),
        )
        .expect("valid");
        let enc = frame.encode();
        assert_eq!(enc.len(), FRAME_HEADER_SIZE + 8);
    }

    #[test]
    fn test_stream_id_msb_cleared_on_encode() {
        // stream_id with MSB set should be cleared
        let frame = Http2Frame {
            length: 0,
            frame_type: Http2FrameType::Data,
            flags: 0,
            stream_id: 0xFFFF_FFFF,
            payload: vec![],
        };
        let enc = frame.encode();
        let decoded_id = u32::from_be_bytes([enc[5], enc[6], enc[7], enc[8]]);
        assert_eq!(decoded_id, 0x7FFF_FFFF, "MSB must be cleared");
    }

    // ── Frame decode ─────────────────────────────────────────────────────────

    #[test]
    fn test_decode_roundtrip_data() {
        let original =
            Http2Frame::new(Http2FrameType::Data, 0, 5, b"body data".to_vec()).expect("valid");
        let enc = original.encode();
        let (decoded, consumed) = Http2Frame::decode(&enc).expect("ok");

        assert_eq!(consumed, enc.len());
        assert_eq!(decoded.frame_type, Http2FrameType::Data);
        assert_eq!(decoded.stream_id, 5);
        assert_eq!(decoded.payload, b"body data");
    }

    #[test]
    fn test_decode_roundtrip_headers() {
        let payload = vec![0x82u8, 0x84, 0x86]; // sample HPACK
        let original = Http2Frame::new(
            Http2FrameType::Headers,
            FLAG_END_HEADERS | FLAG_END_STREAM,
            3,
            payload.clone(),
        )
        .expect("valid");
        let enc = original.encode();
        let (decoded, _) = Http2Frame::decode(&enc).expect("ok");

        assert_eq!(decoded.frame_type, Http2FrameType::Headers);
        assert_eq!(decoded.flags, FLAG_END_HEADERS | FLAG_END_STREAM);
        assert_eq!(decoded.stream_id, 3);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn test_decode_settings_frame() {
        let settings = Http2Settings {
            header_table_size: Some(4096),
            enable_push: Some(0),
            ..Default::default()
        };
        let frame = settings.to_frame().expect("valid");
        let enc = frame.encode();
        let (decoded, _) = Http2Frame::decode(&enc).expect("ok");
        assert_eq!(decoded.frame_type, Http2FrameType::Settings);
        assert_eq!(decoded.stream_id, 0);
    }

    #[test]
    fn test_decode_error_buffer_too_short_header() {
        let result = Http2Frame::decode(&[0x00, 0x00, 0x05, 0x00]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_error_payload_truncated() {
        // Header says payload is 10 bytes but only 3 provided
        let mut bytes = vec![0u8; FRAME_HEADER_SIZE + 3];
        bytes[2] = 10; // length = 10
        let result = Http2Frame::decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_multiple_frames_in_buffer() {
        let f1 = Http2Frame::new(Http2FrameType::Data, 0, 1, b"frame1".to_vec()).expect("ok");
        let f2 = Http2Frame::new(Http2FrameType::Data, FLAG_END_STREAM, 1, b"frame2".to_vec())
            .expect("ok");
        let mut buf = f1.encode();
        buf.extend(f2.encode());

        let (d1, c1) = Http2Frame::decode(&buf).expect("ok");
        let (d2, c2) = Http2Frame::decode(&buf[c1..]).expect("ok");

        assert_eq!(d1.payload, b"frame1");
        assert_eq!(d2.payload, b"frame2");
        assert_eq!(c1 + c2, buf.len());
    }

    #[test]
    fn test_decode_ping_frame() {
        let opaque = [0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let frame = ping_frame(opaque, false).expect("ok");
        let enc = frame.encode();
        let (decoded, _) = Http2Frame::decode(&enc).expect("ok");
        assert_eq!(decoded.frame_type, Http2FrameType::Ping);
        assert_eq!(decoded.payload, opaque);
        assert!(!decoded.has_ack());
    }

    #[test]
    fn test_ping_ack_frame() {
        let frame = ping_frame([0u8; 8], true).expect("ok");
        assert!(frame.has_ack());
    }

    // ── Http2Settings ────────────────────────────────────────────────────────

    #[test]
    fn test_settings_encode_decode_roundtrip() {
        let original = Http2Settings {
            header_table_size: Some(8192),
            enable_push: Some(0),
            max_concurrent_streams: Some(100),
            initial_window_size: Some(65535),
            max_frame_size: Some(16384),
            max_header_list_size: Some(8192),
        };
        let encoded = original.encode();
        let decoded = Http2Settings::decode(&encoded).expect("ok");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_settings_encode_omits_none() {
        let settings = Http2Settings {
            enable_push: Some(0),
            ..Default::default()
        };
        let encoded = settings.encode();
        // Only one parameter → 6 bytes
        assert_eq!(encoded.len(), 6);
        let decoded = Http2Settings::decode(&encoded).expect("ok");
        assert_eq!(decoded.enable_push, Some(0));
        assert!(decoded.header_table_size.is_none());
    }

    #[test]
    fn test_settings_empty_payload() {
        let settings = Http2Settings::decode(&[]).expect("ok");
        assert_eq!(settings, Http2Settings::default());
    }

    #[test]
    fn test_settings_invalid_payload_length() {
        // 7 bytes is not a multiple of 6
        let result = Http2Settings::decode(&[0u8; 7]);
        assert!(result.is_err());
    }

    #[test]
    fn test_settings_unknown_identifier_ignored() {
        // Build a payload with unknown identifier 0xFF
        let mut payload = vec![0x00u8, 0xFF, 0x00, 0x00, 0x00, 0x42];
        // Append a known setting
        payload.extend_from_slice(&[0x00, 0x01, 0x00, 0x00, 0x10, 0x00]); // header_table_size = 4096
        let settings = Http2Settings::decode(&payload).expect("ok");
        assert_eq!(settings.header_table_size, Some(4096));
    }

    #[test]
    fn test_settings_ack_frame() {
        let frame = Http2Settings::ack_frame().expect("ok");
        assert_eq!(frame.frame_type, Http2FrameType::Settings);
        assert!(frame.has_ack());
        assert!(frame.payload.is_empty());
        assert_eq!(frame.stream_id, 0);
    }

    // ── Frame flags ──────────────────────────────────────────────────────────

    #[test]
    fn test_flag_helpers() {
        // FLAG_END_STREAM (0x1) and FLAG_ACK (0x1) intentionally share the same
        // bit value per RFC 7540 — ACK is meaningful only on SETTINGS/PING frames,
        // while END_STREAM is meaningful on DATA/HEADERS frames.
        let frame = Http2Frame::new(
            Http2FrameType::Headers,
            FLAG_END_STREAM | FLAG_END_HEADERS,
            1,
            vec![],
        )
        .expect("ok");
        assert!(frame.has_end_stream());
        assert!(frame.has_end_headers());
        // has_ack() checks bit 0x1 which equals FLAG_END_STREAM; both are true here
        assert!(
            frame.has_ack(),
            "FLAG_ACK == FLAG_END_STREAM == 0x1 per RFC 7540"
        );

        // Verify a frame with only END_HEADERS (0x4) does not report ACK/END_STREAM
        let only_end_headers =
            Http2Frame::new(Http2FrameType::Headers, FLAG_END_HEADERS, 1, vec![]).expect("ok");
        assert!(!only_end_headers.has_end_stream());
        assert!(!only_end_headers.has_ack());
        assert!(only_end_headers.has_end_headers());
    }

    // ── Helper builders ──────────────────────────────────────────────────────

    #[test]
    fn test_window_update_frame() {
        let frame = window_update_frame(0, 65535).expect("ok");
        assert_eq!(frame.frame_type, Http2FrameType::WindowUpdate);
        assert_eq!(frame.payload.len(), 4);
        let inc = u32::from_be_bytes(frame.payload[..4].try_into().expect("4 bytes")) & 0x7FFF_FFFF;
        assert_eq!(inc, 65535);
    }

    #[test]
    fn test_window_update_frame_zero_error() {
        assert!(window_update_frame(0, 0).is_err());
    }

    #[test]
    fn test_rst_stream_frame() {
        let frame = rst_stream_frame(7, 0x8).expect("ok"); // CANCEL
        assert_eq!(frame.frame_type, Http2FrameType::RstStream);
        assert_eq!(frame.stream_id, 7);
        let code = u32::from_be_bytes(frame.payload[..4].try_into().expect("4 bytes"));
        assert_eq!(code, 0x8);
    }

    #[test]
    fn test_goaway_frame() {
        let frame = goaway_frame(10, 0, b"goodbye").expect("ok");
        assert_eq!(frame.frame_type, Http2FrameType::GoAway);
        assert_eq!(frame.stream_id, 0);
        let last_id =
            u32::from_be_bytes(frame.payload[..4].try_into().expect("4 bytes")) & 0x7FFF_FFFF;
        assert_eq!(last_id, 10);
        assert_eq!(&frame.payload[8..], b"goodbye");
    }

    #[test]
    fn test_frame_new_too_large() {
        let large_payload = vec![0u8; (MAX_FRAME_SIZE + 1) as usize];
        let result = Http2Frame::new(Http2FrameType::Data, 0, 1, large_payload);
        assert!(result.is_err());
    }
}
