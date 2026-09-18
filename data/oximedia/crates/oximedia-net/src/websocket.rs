//! WebSocket client implementing RFC 6455.
//!
//! Provides a synchronous WebSocket client with pure Rust HTTP upgrade handshake,
//! RFC 6455 frame encoding/decoding, and client-side masking.

use crate::error::{NetError, NetResult};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use sha1::{Digest, Sha1};
use std::io::{Read, Write};
use std::net::TcpStream;

/// Maximum accepted WebSocket frame payload length.
///
/// RFC 6455 allows a 64-bit length field, so a peer can advertise a payload up
/// to `u64::MAX`. Without a cap, `offset + payload_len` overflows (yielding a
/// start>end slice panic) and the subsequent `to_vec()` would try to allocate
/// an astronomical buffer. 64 MiB is far larger than any control/media frame we
/// exchange.
pub const MAX_WS_FRAME_PAYLOAD: usize = 64 * 1024 * 1024;

/// WebSocket opcode as defined in RFC 6455 §5.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsOpcode {
    /// Continuation frame (opcode 0x0).
    Continuation,
    /// UTF-8 text frame (opcode 0x1).
    Text,
    /// Binary data frame (opcode 0x2).
    Binary,
    /// Connection close frame (opcode 0x8).
    Close,
    /// Ping frame (opcode 0x9).
    Ping,
    /// Pong frame (opcode 0xA).
    Pong,
    /// Unknown/reserved opcode.
    Unknown(u8),
}

impl WsOpcode {
    /// Convert raw nibble to opcode.
    pub fn from_u8(val: u8) -> Self {
        match val & 0x0F {
            0x0 => Self::Continuation,
            0x1 => Self::Text,
            0x2 => Self::Binary,
            0x8 => Self::Close,
            0x9 => Self::Ping,
            0xA => Self::Pong,
            other => Self::Unknown(other),
        }
    }

    /// Convert opcode to raw nibble.
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Continuation => 0x0,
            Self::Text => 0x1,
            Self::Binary => 0x2,
            Self::Close => 0x8,
            Self::Ping => 0x9,
            Self::Pong => 0xA,
            Self::Unknown(v) => v,
        }
    }

    /// Returns true if this is a control frame.
    pub fn is_control(self) -> bool {
        matches!(self, Self::Close | Self::Ping | Self::Pong)
    }
}

/// A WebSocket frame as defined in RFC 6455 §5.
#[derive(Debug, Clone)]
pub struct WsFrame {
    /// Opcode identifying the frame type.
    pub opcode: WsOpcode,
    /// Frame payload data (already unmasked).
    pub payload: Vec<u8>,
    /// FIN bit — true if this is the final fragment.
    pub is_fin: bool,
}

impl WsFrame {
    /// Encode this frame for sending from a client (masked).
    ///
    /// Client → server frames MUST be masked (RFC 6455 §5.3).
    pub fn encode_masked(&self, mask: [u8; 4]) -> Vec<u8> {
        let payload_len = self.payload.len();
        let mut buf = Vec::with_capacity(14 + payload_len);

        // Byte 0: FIN + RSV (0) + opcode
        let b0 = if self.is_fin { 0x80 } else { 0x00 } | self.opcode.as_u8();
        buf.push(b0);

        // Byte 1+: MASK bit set + payload length
        if payload_len < 126 {
            buf.push(0x80 | payload_len as u8);
        } else if payload_len < 65536 {
            buf.push(0x80 | 126u8);
            buf.push((payload_len >> 8) as u8);
            buf.push(payload_len as u8);
        } else {
            buf.push(0x80 | 127u8);
            let len64 = payload_len as u64;
            for i in (0..8).rev() {
                buf.push((len64 >> (i * 8)) as u8);
            }
        }

        // 4-byte masking key
        buf.extend_from_slice(&mask);

        // Masked payload
        for (i, &byte) in self.payload.iter().enumerate() {
            buf.push(byte ^ mask[i % 4]);
        }

        buf
    }

    /// Encode this frame for sending from a server (unmasked).
    pub fn encode_unmasked(&self) -> Vec<u8> {
        let payload_len = self.payload.len();
        let mut buf = Vec::with_capacity(10 + payload_len);

        let b0 = if self.is_fin { 0x80 } else { 0x00 } | self.opcode.as_u8();
        buf.push(b0);

        if payload_len < 126 {
            buf.push(payload_len as u8);
        } else if payload_len < 65536 {
            buf.push(126u8);
            buf.push((payload_len >> 8) as u8);
            buf.push(payload_len as u8);
        } else {
            buf.push(127u8);
            let len64 = payload_len as u64;
            for i in (0..8).rev() {
                buf.push((len64 >> (i * 8)) as u8);
            }
        }

        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Decode a frame from raw bytes.
    ///
    /// Returns the decoded frame and the number of bytes consumed.
    /// Returns `Ok(None)` if the buffer is too short.
    pub fn decode(data: &[u8]) -> NetResult<Option<(Self, usize)>> {
        if data.len() < 2 {
            return Ok(None);
        }

        let b0 = data[0];
        let b1 = data[1];

        let is_fin = (b0 & 0x80) != 0;
        let rsv = (b0 >> 4) & 0x07;
        if rsv != 0 {
            return Err(NetError::protocol(format!(
                "Non-zero RSV bits in WebSocket frame: {rsv:#03x}"
            )));
        }
        let opcode = WsOpcode::from_u8(b0 & 0x0F);
        let is_masked = (b1 & 0x80) != 0;
        let raw_len = (b1 & 0x7F) as usize;

        let mut offset = 2usize;

        // Extended payload length
        let payload_len: usize = if raw_len == 126 {
            if data.len() < offset + 2 {
                return Ok(None);
            }
            let len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;
            len
        } else if raw_len == 127 {
            if data.len() < offset + 8 {
                return Ok(None);
            }
            let bytes: [u8; 8] = data[offset..offset + 8]
                .try_into()
                .map_err(|_| NetError::protocol("Frame length read failed"))?;
            let len = u64::from_be_bytes(bytes) as usize;
            offset += 8;
            len
        } else {
            raw_len
        };

        // Reject an absurd frame length before allocating or slicing: the
        // 64-bit extended length lets a peer request up to u64::MAX bytes,
        // which would overflow `offset + payload_len` and demand a giant alloc.
        if payload_len > MAX_WS_FRAME_PAYLOAD {
            return Err(NetError::protocol(format!(
                "WebSocket frame payload {payload_len} exceeds maximum {MAX_WS_FRAME_PAYLOAD}"
            )));
        }

        // Masking key (4 bytes if mask bit set)
        let mask: Option<[u8; 4]> = if is_masked {
            if data.len() < offset + 4 {
                return Ok(None);
            }
            let key = [
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ];
            offset += 4;
            Some(key)
        } else {
            None
        };

        // Payload. `checked_add` guards the (already capped) length against any
        // residual overflow before slicing.
        let payload_end = offset
            .checked_add(payload_len)
            .ok_or_else(|| NetError::protocol("WebSocket frame length overflow"))?;
        if data.len() < payload_end {
            return Ok(None);
        }
        let mut payload = data[offset..payload_end].to_vec();
        offset = payload_end;

        // Unmask if needed
        if let Some(key) = mask {
            for (i, byte) in payload.iter_mut().enumerate() {
                *byte ^= key[i % 4];
            }
        }

        Ok(Some((
            WsFrame {
                opcode,
                payload,
                is_fin,
            },
            offset,
        )))
    }
}

/// WebSocket client state.
#[derive(Debug, PartialEq, Eq)]
enum WsState {
    Open,
    Closing,
    Closed,
}

/// Synchronous WebSocket client (RFC 6455).
pub struct WsClient {
    stream: TcpStream,
    read_buf: Vec<u8>,
    state: WsState,
    /// Mask seed — incremented for each frame to produce distinct masks.
    mask_counter: u32,
}

impl WsClient {
    /// Connect to a WebSocket server at the given `ws://` or `wss://` URL.
    ///
    /// Performs TCP connection and HTTP/1.1 Upgrade handshake.
    /// Note: `wss://` (TLS) is not currently supported; use `ws://` only.
    pub fn connect(url: &str) -> NetResult<Self> {
        let (host, port, path) = parse_ws_url(url)?;

        let addr = format!("{host}:{port}");
        let stream = TcpStream::connect(&addr)
            .map_err(|e| NetError::connection(format!("TCP connect to {addr} failed: {e}")))?;

        let mut client = WsClient {
            stream,
            read_buf: Vec::new(),
            state: WsState::Open,
            mask_counter: 0x12345678,
        };

        client.perform_handshake(&host, port, &path)?;
        Ok(client)
    }

    /// Perform the HTTP/1.1 WebSocket upgrade handshake.
    fn perform_handshake(&mut self, host: &str, port: u16, path: &str) -> NetResult<()> {
        // Generate a random 16-byte nonce and base64-encode it
        let nonce = generate_nonce(self.mask_counter);
        self.mask_counter = self.mask_counter.wrapping_add(1);
        let key = B64.encode(nonce);

        let request = format!(
            "GET {path} HTTP/1.1\r\n\
             Host: {host}:{port}\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Key: {key}\r\n\
             Sec-WebSocket-Version: 13\r\n\
             \r\n"
        );

        self.stream
            .write_all(request.as_bytes())
            .map_err(|e| NetError::Io(e))?;

        // Read response headers
        let response = read_http_response(&mut self.stream)?;

        // Verify 101 Switching Protocols
        if !response.starts_with("HTTP/1.1 101") {
            return Err(NetError::handshake(format!(
                "Expected 101 Switching Protocols, got: {}",
                response.lines().next().unwrap_or("(empty)")
            )));
        }

        // Verify Sec-WebSocket-Accept
        let expected_accept = compute_accept_key(&key);
        let accept_header = find_header(&response, "Sec-WebSocket-Accept:");
        match accept_header {
            Some(val) if val.trim() == expected_accept => {}
            Some(val) => {
                return Err(NetError::handshake(format!(
                    "Sec-WebSocket-Accept mismatch: got '{val}', expected '{expected_accept}'"
                )));
            }
            None => {
                return Err(NetError::handshake(
                    "Missing Sec-WebSocket-Accept header".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Send a text message (UTF-8).
    pub fn send_text(&mut self, text: &str) -> NetResult<()> {
        self.send_frame(WsOpcode::Text, text.as_bytes())
    }

    /// Send a binary message.
    pub fn send_binary(&mut self, data: &[u8]) -> NetResult<()> {
        self.send_frame(WsOpcode::Binary, data)
    }

    /// Send a Ping frame with optional payload.
    pub fn send_ping(&mut self, payload: &[u8]) -> NetResult<()> {
        self.send_frame(WsOpcode::Ping, payload)
    }

    /// Send a Close frame initiating connection teardown.
    pub fn send_close(&mut self, code: u16, reason: &str) -> NetResult<()> {
        let mut payload = Vec::with_capacity(2 + reason.len());
        payload.push((code >> 8) as u8);
        payload.push(code as u8);
        payload.extend_from_slice(reason.as_bytes());
        self.state = WsState::Closing;
        self.send_frame(WsOpcode::Close, &payload)
    }

    /// Send a single complete (FIN=1) masked frame.
    fn send_frame(&mut self, opcode: WsOpcode, payload: &[u8]) -> NetResult<()> {
        if self.state == WsState::Closed {
            return Err(NetError::invalid_state("WebSocket is closed"));
        }
        let mask = self.next_mask();
        let frame = WsFrame {
            opcode,
            payload: payload.to_vec(),
            is_fin: true,
        };
        let encoded = frame.encode_masked(mask);
        self.stream
            .write_all(&encoded)
            .map_err(|e| NetError::Io(e))?;
        Ok(())
    }

    /// Receive the next complete WebSocket frame.
    ///
    /// Automatically responds to Ping frames with a Pong.
    /// Returns `Ok(None)` when the connection is closed gracefully.
    pub fn recv_frame(&mut self) -> NetResult<Option<WsFrame>> {
        loop {
            // Try to decode a frame from the buffer
            if !self.read_buf.is_empty() {
                match WsFrame::decode(&self.read_buf)? {
                    Some((frame, consumed)) => {
                        self.read_buf.drain(..consumed);

                        // Handle control frames automatically
                        match frame.opcode {
                            WsOpcode::Ping => {
                                // Auto-respond with Pong
                                self.send_frame(WsOpcode::Pong, &frame.payload)?;
                                continue;
                            }
                            WsOpcode::Close => {
                                self.state = WsState::Closed;
                                return Ok(None);
                            }
                            _ => return Ok(Some(frame)),
                        }
                    }
                    None => {
                        // Need more data — fall through to read
                    }
                }
            }

            if self.state == WsState::Closed {
                return Ok(None);
            }

            // Read more data from the socket
            let mut tmp = [0u8; 4096];
            let n = self.stream.read(&mut tmp).map_err(|e| NetError::Io(e))?;
            if n == 0 {
                self.state = WsState::Closed;
                return Ok(None);
            }
            self.read_buf.extend_from_slice(&tmp[..n]);
        }
    }

    /// Receive a complete message, reassembling fragmented frames.
    ///
    /// Returns (opcode, reassembled_payload) for the logical message.
    pub fn recv_message(&mut self) -> NetResult<Option<(WsOpcode, Vec<u8>)>> {
        let mut message_opcode: Option<WsOpcode> = None;
        let mut payload = Vec::new();

        loop {
            let frame = match self.recv_frame()? {
                Some(f) => f,
                None => return Ok(None),
            };

            match frame.opcode {
                WsOpcode::Continuation => {
                    payload.extend_from_slice(&frame.payload);
                    if frame.is_fin {
                        let opcode = message_opcode.take().unwrap_or(WsOpcode::Binary);
                        return Ok(Some((opcode, payload)));
                    }
                }
                WsOpcode::Text | WsOpcode::Binary => {
                    if frame.is_fin {
                        return Ok(Some((frame.opcode, frame.payload)));
                    }
                    message_opcode = Some(frame.opcode);
                    payload = frame.payload;
                }
                _ => {
                    // Control frames are handled in recv_frame; shouldn't reach here
                    continue;
                }
            }
        }
    }

    /// Generate the next 4-byte masking key using a simple counter-based PRNG.
    fn next_mask(&mut self) -> [u8; 4] {
        // xorshift32 — fast, deterministic, sufficient for RFC 6455 masking
        let mut x = self.mask_counter;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.mask_counter = x;
        x.to_ne_bytes()
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Parse a `ws://host[:port]/path` URL into (host, port, path).
fn parse_ws_url(url: &str) -> NetResult<(String, u16, String)> {
    let rest = if let Some(r) = url.strip_prefix("ws://") {
        r
    } else if let Some(r) = url.strip_prefix("wss://") {
        r
    } else {
        return Err(NetError::invalid_url(format!(
            "URL must start with ws:// or wss://: {url}"
        )));
    };

    // Split host/port from path
    let (authority, path) = if let Some(slash) = rest.find('/') {
        (&rest[..slash], rest[slash..].to_string())
    } else {
        (rest, "/".to_string())
    };

    // Split host and optional port
    let (host, port) = if let Some(colon) = authority.rfind(':') {
        let port_str = &authority[colon + 1..];
        let port: u16 = port_str
            .parse()
            .map_err(|_| NetError::invalid_url(format!("Invalid port in URL: {port_str}")))?;
        (authority[..colon].to_string(), port)
    } else {
        let default_port = if url.starts_with("wss://") { 443 } else { 80 };
        (authority.to_string(), default_port)
    };

    if host.is_empty() {
        return Err(NetError::invalid_url(
            "Empty host in WebSocket URL".to_string(),
        ));
    }

    Ok((host, port, path))
}

/// Generate a 16-byte nonce from a counter seed.
fn generate_nonce(seed: u32) -> [u8; 16] {
    let mut state = seed.wrapping_add(0xDEADBEEF);
    let mut out = [0u8; 16];
    for chunk in out.chunks_mut(4) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let bytes = state.to_le_bytes();
        let len = chunk.len();
        chunk.copy_from_slice(&bytes[..len]);
    }
    out
}

/// Compute the Sec-WebSocket-Accept response from the client key.
fn compute_accept_key(client_key: &str) -> String {
    const WS_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    let combined = format!("{client_key}{WS_GUID}");
    let mut hasher = Sha1::new();
    hasher.update(combined.as_bytes());
    let hash = hasher.finalize();
    B64.encode(hash)
}

/// Read an HTTP response from a `TcpStream` until `\r\n\r\n`.
fn read_http_response(stream: &mut TcpStream) -> NetResult<String> {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];

    loop {
        stream.read_exact(&mut byte).map_err(|e| NetError::Io(e))?;
        buf.push(byte[0]);

        if buf.ends_with(b"\r\n\r\n") {
            break;
        }
        if buf.len() > 16 * 1024 {
            return Err(NetError::protocol("HTTP response headers too large"));
        }
    }

    String::from_utf8(buf).map_err(|_| NetError::protocol("HTTP response is not valid UTF-8"))
}

/// Find the value of an HTTP response header (case-insensitive name match).
fn find_header<'a>(response: &'a str, name: &str) -> Option<&'a str> {
    let name_lower = name.to_lowercase();
    for line in response.lines() {
        let lower = line.to_lowercase();
        if lower.starts_with(&name_lower) {
            let value = &line[name.len()..];
            return Some(value.trim());
        }
    }
    None
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── WsOpcode ────────────────────────────────────────────────────────────

    #[test]
    fn test_opcode_roundtrip() {
        for &op in &[
            WsOpcode::Continuation,
            WsOpcode::Text,
            WsOpcode::Binary,
            WsOpcode::Close,
            WsOpcode::Ping,
            WsOpcode::Pong,
        ] {
            assert_eq!(WsOpcode::from_u8(op.as_u8()), op);
        }
    }

    #[test]
    fn test_opcode_unknown() {
        let op = WsOpcode::from_u8(0x3);
        assert!(matches!(op, WsOpcode::Unknown(3)));
        assert_eq!(op.as_u8(), 3);
    }

    #[test]
    fn test_opcode_is_control() {
        assert!(WsOpcode::Close.is_control());
        assert!(WsOpcode::Ping.is_control());
        assert!(WsOpcode::Pong.is_control());
        assert!(!WsOpcode::Text.is_control());
        assert!(!WsOpcode::Binary.is_control());
        assert!(!WsOpcode::Continuation.is_control());
    }

    // ── Frame encoding ───────────────────────────────────────────────────────

    #[test]
    fn test_encode_small_text_masked() {
        let frame = WsFrame {
            opcode: WsOpcode::Text,
            payload: b"Hello".to_vec(),
            is_fin: true,
        };
        let mask = [0x37, 0xfa, 0x21, 0x3d];
        let encoded = frame.encode_masked(mask);

        // FIN=1, opcode=1
        assert_eq!(encoded[0], 0x81);
        // MASK=1, len=5
        assert_eq!(encoded[1], 0x85);
        // masking key
        assert_eq!(&encoded[2..6], &mask);
        // payload is masked
        assert_ne!(&encoded[6..], b"Hello");
    }

    #[test]
    fn test_encode_decode_roundtrip_small() {
        let original_payload = b"Hello, WebSocket!".to_vec();
        let frame = WsFrame {
            opcode: WsOpcode::Text,
            payload: original_payload.clone(),
            is_fin: true,
        };
        let mask = [0xAA, 0xBB, 0xCC, 0xDD];
        let encoded = frame.encode_masked(mask);

        // Decode (client sends masked; simulate server receiving it)
        let (decoded, consumed) = WsFrame::decode(&encoded)
            .expect("decode ok")
            .expect("complete frame");

        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.payload, original_payload);
        assert!(decoded.is_fin);
        assert_eq!(decoded.opcode, WsOpcode::Text);
    }

    #[test]
    fn test_encode_decode_roundtrip_unmasked() {
        let payload = b"server response".to_vec();
        let frame = WsFrame {
            opcode: WsOpcode::Binary,
            payload: payload.clone(),
            is_fin: true,
        };
        let encoded = frame.encode_unmasked();

        let (decoded, consumed) = WsFrame::decode(&encoded)
            .expect("decode ok")
            .expect("frame present");

        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.payload, payload);
        assert_eq!(decoded.opcode, WsOpcode::Binary);
    }

    #[test]
    fn test_frame_decode_insufficient_data() {
        // Only 1 byte — need at least 2
        let result = WsFrame::decode(&[0x81]).expect("no error");
        assert!(result.is_none());
    }

    #[test]
    fn test_frame_decode_incomplete_payload() {
        // Header says 10 bytes but only 2 present
        let data = [0x81u8, 0x0A, 0x01, 0x02]; // FIN+Text, len=10, only 2 bytes of payload
        let result = WsFrame::decode(&data).expect("no error");
        assert!(result.is_none());
    }

    #[test]
    fn frame_decode_gigantic_length_is_rejected_not_panic() {
        // FIN+binary, 64-bit extended length = u64::MAX, no mask, no payload.
        // `offset + payload_len` would overflow and panic on the slice; the
        // decoder must reject it cleanly.
        let mut data = vec![0x82u8, 0x7F];
        data.extend_from_slice(&u64::MAX.to_be_bytes());
        let result = WsFrame::decode(&data);
        assert!(result.is_err(), "gigantic frame length must be an error");
    }

    #[test]
    fn frame_decode_over_cap_length_is_rejected() {
        // A length that fits in usize but exceeds MAX_WS_FRAME_PAYLOAD.
        let over = (MAX_WS_FRAME_PAYLOAD as u64) + 1;
        let mut data = vec![0x82u8, 0x7F];
        data.extend_from_slice(&over.to_be_bytes());
        assert!(WsFrame::decode(&data).is_err());
    }

    #[test]
    fn test_frame_decode_16bit_length() {
        // Build a frame with 126-byte payload
        let payload = vec![0x42u8; 200];
        let frame = WsFrame {
            opcode: WsOpcode::Binary,
            payload: payload.clone(),
            is_fin: false,
        };
        let encoded = frame.encode_unmasked();

        // Byte 1 should be 126 (two-byte extended length, no mask)
        assert_eq!(encoded[1], 126);
        let (decoded, _) = WsFrame::decode(&encoded).expect("ok").expect("complete");
        assert_eq!(decoded.payload, payload);
        assert!(!decoded.is_fin);
    }

    #[test]
    fn test_frame_decode_64bit_length() {
        // Payload exactly at 64k boundary
        let payload = vec![0xFFu8; 65536];
        let frame = WsFrame {
            opcode: WsOpcode::Binary,
            payload: payload.clone(),
            is_fin: true,
        };
        let encoded = frame.encode_unmasked();
        // Byte 1 should be 127 (8-byte extended length)
        assert_eq!(encoded[1], 127);
        let (decoded, consumed) = WsFrame::decode(&encoded).expect("ok").expect("complete");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.payload.len(), 65536);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn test_frame_rsv_bits_error() {
        // Byte 0 with RSV1 set (0x40)
        let data = [0xC1u8, 0x01, 0xFF]; // FIN+RSV1+Text, len=1
        let result = WsFrame::decode(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_close_frame_opcode() {
        let frame = WsFrame {
            opcode: WsOpcode::Close,
            payload: vec![0x03, 0xE8], // status 1000
            is_fin: true,
        };
        let encoded = frame.encode_unmasked();
        let (decoded, _) = WsFrame::decode(&encoded).expect("ok").expect("frame");
        assert_eq!(decoded.opcode, WsOpcode::Close);
        assert_eq!(decoded.payload, vec![0x03, 0xE8]);
    }

    #[test]
    fn test_ping_pong_frames() {
        for opcode in [WsOpcode::Ping, WsOpcode::Pong] {
            let frame = WsFrame {
                opcode,
                payload: b"ping data".to_vec(),
                is_fin: true,
            };
            let encoded = frame.encode_unmasked();
            let (decoded, _) = WsFrame::decode(&encoded).expect("ok").expect("frame");
            assert_eq!(decoded.opcode, opcode);
            assert_eq!(decoded.payload, b"ping data");
        }
    }

    #[test]
    fn test_empty_payload_frame() {
        let frame = WsFrame {
            opcode: WsOpcode::Ping,
            payload: vec![],
            is_fin: true,
        };
        let encoded = frame.encode_unmasked();
        assert_eq!(encoded.len(), 2); // header only
        let (decoded, consumed) = WsFrame::decode(&encoded).expect("ok").expect("frame");
        assert_eq!(consumed, 2);
        assert!(decoded.payload.is_empty());
    }

    // ── URL parsing ──────────────────────────────────────────────────────────

    #[test]
    fn test_parse_ws_url_simple() {
        let (host, port, path) = parse_ws_url("ws://example.com/chat").expect("ok");
        assert_eq!(host, "example.com");
        assert_eq!(port, 80);
        assert_eq!(path, "/chat");
    }

    #[test]
    fn test_parse_ws_url_with_port() {
        let (host, port, path) = parse_ws_url("ws://localhost:9001/").expect("ok");
        assert_eq!(host, "localhost");
        assert_eq!(port, 9001);
        assert_eq!(path, "/");
    }

    #[test]
    fn test_parse_wss_url_default_port() {
        let (host, port, path) = parse_ws_url("wss://secure.example.com/ws").expect("ok");
        assert_eq!(host, "secure.example.com");
        assert_eq!(port, 443);
        assert_eq!(path, "/ws");
    }

    #[test]
    fn test_parse_ws_url_no_path() {
        let (host, port, path) = parse_ws_url("ws://example.com").expect("ok");
        assert_eq!(host, "example.com");
        assert_eq!(port, 80);
        assert_eq!(path, "/");
    }

    #[test]
    fn test_parse_ws_url_invalid_scheme() {
        let result = parse_ws_url("http://example.com/ws");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_ws_url_invalid_port() {
        let result = parse_ws_url("ws://example.com:notaport/ws");
        assert!(result.is_err());
    }

    // ── Sec-WebSocket-Accept ─────────────────────────────────────────────────

    #[test]
    fn test_compute_accept_key_rfc_example() {
        // RFC 6455 §1.3 example
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        let accept = compute_accept_key(key);
        assert_eq!(accept, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    // ── find_header ──────────────────────────────────────────────────────────

    #[test]
    fn test_find_header_present() {
        let response = "HTTP/1.1 101 Switching Protocols\r\nSec-WebSocket-Accept: abc123==\r\n\r\n";
        let val = find_header(response, "Sec-WebSocket-Accept:");
        assert_eq!(val, Some("abc123=="));
    }

    #[test]
    fn test_find_header_absent() {
        let response = "HTTP/1.1 101 Switching Protocols\r\n\r\n";
        let val = find_header(response, "Sec-WebSocket-Accept:");
        assert!(val.is_none());
    }

    // ── Masking ──────────────────────────────────────────────────────────────

    #[test]
    fn test_masking_xor_correctness() {
        let payload = b"Test payload data for masking verification";
        let mask = [0x12u8, 0x34, 0x56, 0x78];

        let frame = WsFrame {
            opcode: WsOpcode::Binary,
            payload: payload.to_vec(),
            is_fin: true,
        };
        let encoded = frame.encode_masked(mask);

        // Decode to get unmasked payload back
        let (decoded, _) = WsFrame::decode(&encoded).expect("ok").expect("frame");
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn test_mask_applied_symmetrically() {
        // XOR masking must be its own inverse
        let data = b"symmetric test";
        let mask = [0xAB, 0xCD, 0xEF, 0x01];
        let masked: Vec<u8> = data
            .iter()
            .enumerate()
            .map(|(i, &b)| b ^ mask[i % 4])
            .collect();
        let unmasked: Vec<u8> = masked
            .iter()
            .enumerate()
            .map(|(i, &b)| b ^ mask[i % 4])
            .collect();
        assert_eq!(unmasked, data);
    }

    // ── generate_nonce ───────────────────────────────────────────────────────

    #[test]
    fn test_nonce_is_16_bytes() {
        let nonce = generate_nonce(42);
        assert_eq!(nonce.len(), 16);
    }

    #[test]
    fn test_nonce_different_seeds_differ() {
        let n1 = generate_nonce(1);
        let n2 = generate_nonce(2);
        assert_ne!(n1, n2);
    }
}
