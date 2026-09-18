//! RTMP chunk stream handling.
//!
//! RTMP messages are split into chunks for multiplexing multiple streams
//! over a single connection.

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
use std::collections::HashMap;

/// Default chunk size (128 bytes).
pub const DEFAULT_CHUNK_SIZE: u32 = 128;

/// Maximum chunk size.
pub const MAX_CHUNK_SIZE: u32 = 65536;

/// Upper bound on the buffer we pre-allocate for a reassembling message.
///
/// A chunk's `message_length` is a 24-bit field (up to 16 MiB) and there is a
/// separate reassembly buffer per chunk-stream-id (up to ~65k of them). Trusting
/// the declared length would let a peer send thousands of tiny chunk headers,
/// each declaring 16 MiB, to force ~1 TB of reservation before sending any
/// payload. We instead pre-allocate at most this much and let the buffer grow
/// as real bytes arrive, so memory tracks delivered data, not declared intent.
const MAX_MESSAGE_PREALLOC: usize = 64 * 1024;

/// Chunk header types (format field).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkHeaderType {
    /// Type 0: Full header (11 bytes).
    Full,
    /// Type 1: Same stream ID (7 bytes).
    SameStream,
    /// Type 2: Same stream ID and message length (3 bytes).
    TimestampOnly,
    /// Type 3: No header (continuation).
    Continuation,
}

impl ChunkHeaderType {
    /// Returns the format value (0-3).
    #[must_use]
    pub const fn format_value(&self) -> u8 {
        match self {
            Self::Full => 0,
            Self::SameStream => 1,
            Self::TimestampOnly => 2,
            Self::Continuation => 3,
        }
    }

    /// Creates from format value.
    #[must_use]
    pub const fn from_format(fmt: u8) -> Option<Self> {
        match fmt {
            0 => Some(Self::Full),
            1 => Some(Self::SameStream),
            2 => Some(Self::TimestampOnly),
            3 => Some(Self::Continuation),
            _ => None,
        }
    }

    /// Returns the header size in bytes (excluding basic header).
    #[must_use]
    pub const fn header_size(&self) -> usize {
        match self {
            Self::Full => 11,
            Self::SameStream => 7,
            Self::TimestampOnly => 3,
            Self::Continuation => 0,
        }
    }
}

/// Message header information.
#[derive(Debug, Clone, Copy, Default)]
pub struct MessageHeader {
    /// Timestamp (delta or absolute).
    pub timestamp: u32,
    /// Message length in bytes.
    pub message_length: u32,
    /// Message type ID.
    pub message_type: u8,
    /// Message stream ID.
    pub message_stream_id: u32,
}

impl MessageHeader {
    /// Creates a new message header.
    #[must_use]
    pub const fn new(
        timestamp: u32,
        message_length: u32,
        message_type: u8,
        message_stream_id: u32,
    ) -> Self {
        Self {
            timestamp,
            message_length,
            message_type,
            message_stream_id,
        }
    }

    /// Returns true if timestamp needs extended format.
    #[must_use]
    pub const fn needs_extended_timestamp(&self) -> bool {
        self.timestamp >= 0x00FF_FFFF
    }
}

/// Chunk header (basic + message header).
#[derive(Debug, Clone)]
pub struct ChunkHeader {
    /// Header format type.
    pub format: ChunkHeaderType,
    /// Chunk stream ID (2-65599).
    pub chunk_stream_id: u32,
    /// Message header.
    pub message: MessageHeader,
    /// Extended timestamp (if timestamp >= 0xFFFFFF).
    pub extended_timestamp: Option<u32>,
}

impl ChunkHeader {
    /// Creates a new chunk header.
    #[must_use]
    pub fn new(format: ChunkHeaderType, chunk_stream_id: u32, message: MessageHeader) -> Self {
        let extended_timestamp = if message.needs_extended_timestamp() {
            Some(message.timestamp)
        } else {
            None
        };

        Self {
            format,
            chunk_stream_id,
            message,
            extended_timestamp,
        }
    }

    /// Returns the actual timestamp value.
    #[must_use]
    pub const fn timestamp(&self) -> u32 {
        match self.extended_timestamp {
            Some(ts) => ts,
            None => self.message.timestamp,
        }
    }

    /// Encodes the basic header.
    pub fn encode_basic_header(&self, buf: &mut BytesMut) {
        let fmt = self.format.format_value();
        let csid = self.chunk_stream_id;

        if csid < 64 {
            // 1 byte format
            buf.put_u8((fmt << 6) | (csid as u8));
        } else if csid < 320 {
            // 2 byte format (csid - 64)
            buf.put_u8(fmt << 6);
            buf.put_u8((csid - 64) as u8);
        } else {
            // 3 byte format (csid - 64)
            buf.put_u8((fmt << 6) | 1);
            let adjusted = csid - 64;
            buf.put_u8((adjusted & 0xFF) as u8);
            buf.put_u8(((adjusted >> 8) & 0xFF) as u8);
        }
    }

    /// Encodes the full chunk header.
    pub fn encode(&self, buf: &mut BytesMut) {
        self.encode_basic_header(buf);

        let timestamp = if self.message.needs_extended_timestamp() {
            0x00FF_FFFF
        } else {
            self.message.timestamp
        };

        match self.format {
            ChunkHeaderType::Full => {
                // Timestamp (3 bytes)
                buf.put_u8(((timestamp >> 16) & 0xFF) as u8);
                buf.put_u8(((timestamp >> 8) & 0xFF) as u8);
                buf.put_u8((timestamp & 0xFF) as u8);
                // Message length (3 bytes)
                buf.put_u8(((self.message.message_length >> 16) & 0xFF) as u8);
                buf.put_u8(((self.message.message_length >> 8) & 0xFF) as u8);
                buf.put_u8((self.message.message_length & 0xFF) as u8);
                // Message type (1 byte)
                buf.put_u8(self.message.message_type);
                // Message stream ID (4 bytes, little endian)
                buf.put_u32_le(self.message.message_stream_id);
            }
            ChunkHeaderType::SameStream => {
                // Timestamp delta (3 bytes)
                buf.put_u8(((timestamp >> 16) & 0xFF) as u8);
                buf.put_u8(((timestamp >> 8) & 0xFF) as u8);
                buf.put_u8((timestamp & 0xFF) as u8);
                // Message length (3 bytes)
                buf.put_u8(((self.message.message_length >> 16) & 0xFF) as u8);
                buf.put_u8(((self.message.message_length >> 8) & 0xFF) as u8);
                buf.put_u8((self.message.message_length & 0xFF) as u8);
                // Message type (1 byte)
                buf.put_u8(self.message.message_type);
            }
            ChunkHeaderType::TimestampOnly => {
                // Timestamp delta (3 bytes)
                buf.put_u8(((timestamp >> 16) & 0xFF) as u8);
                buf.put_u8(((timestamp >> 8) & 0xFF) as u8);
                buf.put_u8((timestamp & 0xFF) as u8);
            }
            ChunkHeaderType::Continuation => {
                // No message header
            }
        }

        // Extended timestamp
        if let Some(ext) = self.extended_timestamp {
            buf.put_u32(ext);
        }
    }

    /// Decodes a chunk header from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the header is malformed.
    pub fn decode(buf: &mut &[u8], prev_headers: &HashMap<u32, ChunkHeader>) -> NetResult<Self> {
        if buf.is_empty() {
            return Err(NetError::parse(0, "Empty chunk header"));
        }

        // Parse basic header
        let first_byte = buf.get_u8();
        let fmt = first_byte >> 6;
        let csid_marker = first_byte & 0x3F;

        let chunk_stream_id = match csid_marker {
            0 => {
                if buf.is_empty() {
                    return Err(NetError::parse(0, "Incomplete chunk stream ID"));
                }
                64 + u32::from(buf.get_u8())
            }
            1 => {
                if buf.len() < 2 {
                    return Err(NetError::parse(0, "Incomplete chunk stream ID"));
                }
                let b0 = buf.get_u8();
                let b1 = buf.get_u8();
                64 + u32::from(b0) + (u32::from(b1) << 8)
            }
            _ => u32::from(csid_marker),
        };

        let format = ChunkHeaderType::from_format(fmt)
            .ok_or_else(|| NetError::parse(0, "Invalid chunk format"))?;

        // Get previous header for this chunk stream
        let prev = prev_headers.get(&chunk_stream_id);

        // Parse message header based on format
        let mut message = prev.map(|h| h.message).unwrap_or_default();
        #[allow(unused_assignments)]
        let mut timestamp_field = 0u32;

        match format {
            ChunkHeaderType::Full => {
                if buf.len() < 11 {
                    return Err(NetError::parse(0, "Incomplete type 0 header"));
                }
                // Timestamp (3 bytes, big endian)
                timestamp_field = (u32::from(buf.get_u8()) << 16)
                    | (u32::from(buf.get_u8()) << 8)
                    | u32::from(buf.get_u8());
                // Message length (3 bytes)
                message.message_length = (u32::from(buf.get_u8()) << 16)
                    | (u32::from(buf.get_u8()) << 8)
                    | u32::from(buf.get_u8());
                // Message type
                message.message_type = buf.get_u8();
                // Message stream ID (4 bytes, little endian)
                message.message_stream_id = buf.get_u32_le();
                message.timestamp = timestamp_field;
            }
            ChunkHeaderType::SameStream => {
                if buf.len() < 7 {
                    return Err(NetError::parse(0, "Incomplete type 1 header"));
                }
                timestamp_field = (u32::from(buf.get_u8()) << 16)
                    | (u32::from(buf.get_u8()) << 8)
                    | u32::from(buf.get_u8());
                message.message_length = (u32::from(buf.get_u8()) << 16)
                    | (u32::from(buf.get_u8()) << 8)
                    | u32::from(buf.get_u8());
                message.message_type = buf.get_u8();
                message.timestamp = timestamp_field;
            }
            ChunkHeaderType::TimestampOnly => {
                if buf.len() < 3 {
                    return Err(NetError::parse(0, "Incomplete type 2 header"));
                }
                timestamp_field = (u32::from(buf.get_u8()) << 16)
                    | (u32::from(buf.get_u8()) << 8)
                    | u32::from(buf.get_u8());
                message.timestamp = timestamp_field;
            }
            ChunkHeaderType::Continuation => {
                // Use previous timestamp
                timestamp_field = prev.map(|h| h.message.timestamp).unwrap_or(0);
            }
        }

        // Extended timestamp
        let extended_timestamp = if timestamp_field == 0x00FF_FFFF {
            if buf.len() < 4 {
                return Err(NetError::parse(0, "Incomplete extended timestamp"));
            }
            let ext = buf.get_u32();
            message.timestamp = ext;
            Some(ext)
        } else {
            None
        };

        Ok(Self {
            format,
            chunk_stream_id,
            message,
            extended_timestamp,
        })
    }
}

/// Chunk stream state for message reassembly.
#[derive(Debug)]
struct ChunkStreamState {
    /// Last header for this chunk stream.
    last_header: Option<ChunkHeader>,
    /// Accumulated message data.
    message_buffer: BytesMut,
    /// Expected message length.
    message_length: u32,
    /// Bytes received for current message.
    bytes_received: u32,
}

impl ChunkStreamState {
    fn new() -> Self {
        Self {
            last_header: None,
            message_buffer: BytesMut::new(),
            message_length: 0,
            bytes_received: 0,
        }
    }

    fn reset(&mut self) {
        self.message_buffer.clear();
        self.message_length = 0;
        self.bytes_received = 0;
    }
}

/// Assembled RTMP message.
#[derive(Debug, Clone)]
pub struct AssembledMessage {
    /// Chunk stream ID.
    pub chunk_stream_id: u32,
    /// Message header.
    pub header: MessageHeader,
    /// Message payload.
    pub payload: Bytes,
}

/// Chunk stream multiplexer/demultiplexer.
#[derive(Debug)]
pub struct ChunkStream {
    /// Receive chunk size.
    rx_chunk_size: u32,
    /// Transmit chunk size.
    tx_chunk_size: u32,
    /// Chunk stream states.
    streams: HashMap<u32, ChunkStreamState>,
    /// Previous headers for encoding.
    prev_headers: HashMap<u32, ChunkHeader>,
}

impl ChunkStream {
    /// Creates a new chunk stream handler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rx_chunk_size: DEFAULT_CHUNK_SIZE,
            tx_chunk_size: DEFAULT_CHUNK_SIZE,
            streams: HashMap::new(),
            prev_headers: HashMap::new(),
        }
    }

    /// Sets the receive chunk size.
    pub fn set_rx_chunk_size(&mut self, size: u32) {
        self.rx_chunk_size = size.min(MAX_CHUNK_SIZE);
    }

    /// Sets the transmit chunk size.
    pub fn set_tx_chunk_size(&mut self, size: u32) {
        self.tx_chunk_size = size.min(MAX_CHUNK_SIZE);
    }

    /// Returns the current receive chunk size.
    #[must_use]
    pub const fn rx_chunk_size(&self) -> u32 {
        self.rx_chunk_size
    }

    /// Returns the current transmit chunk size.
    #[must_use]
    pub const fn tx_chunk_size(&self) -> u32 {
        self.tx_chunk_size
    }

    /// Processes incoming chunk data.
    ///
    /// Returns assembled messages when complete.
    ///
    /// # Errors
    ///
    /// Returns an error if the chunk data is malformed.
    pub fn process_chunk(&mut self, data: &[u8]) -> NetResult<Vec<AssembledMessage>> {
        let mut messages = Vec::new();
        let mut buf = data;

        while !buf.is_empty() {
            // Decode chunk header
            let header = ChunkHeader::decode(&mut buf, &self.prev_headers)?;
            let csid = header.chunk_stream_id;

            // Get or create stream state
            let state = self
                .streams
                .entry(csid)
                .or_insert_with(ChunkStreamState::new);

            // Start new message if this is a new message header
            if !matches!(header.format, ChunkHeaderType::Continuation) || state.message_length == 0
            {
                state.reset();
                state.message_length = header.message.message_length;
                // Bound the pre-allocation: never reserve the full declared
                // length (a 16 MiB claim per chunk-stream-id would amplify a
                // few bytes of headers into ~1 TB). The buffer grows via
                // `put_slice` as genuine payload arrives.
                let prealloc = (state.message_length as usize).min(MAX_MESSAGE_PREALLOC);
                state.message_buffer.reserve(prealloc);
            }

            // Calculate chunk payload size. `saturating_sub` guards against a
            // malformed continuation that reports more bytes received than the
            // declared length (which would panic on a plain subtraction).
            let remaining = state.message_length.saturating_sub(state.bytes_received);
            let chunk_size = remaining.min(self.rx_chunk_size);

            if buf.len() < chunk_size as usize {
                return Err(NetError::parse(0, "Incomplete chunk payload"));
            }

            // Copy chunk payload
            state.message_buffer.put_slice(&buf[..chunk_size as usize]);
            state.bytes_received += chunk_size;
            buf = &buf[chunk_size as usize..];

            // Check if message is complete
            if state.bytes_received >= state.message_length {
                messages.push(AssembledMessage {
                    chunk_stream_id: csid,
                    header: header.message,
                    payload: state.message_buffer.split().freeze(),
                });
                state.reset();
            }

            // Save header for delta encoding
            state.last_header = Some(header.clone());
            self.prev_headers.insert(csid, header);
        }

        Ok(messages)
    }

    /// Encodes a message into chunks.
    #[must_use]
    pub fn encode_message(
        &mut self,
        chunk_stream_id: u32,
        message: &MessageHeader,
        payload: &[u8],
    ) -> Bytes {
        let mut buf = BytesMut::new();

        // Determine header type based on previous message
        let format = match self.prev_headers.get(&chunk_stream_id) {
            None => ChunkHeaderType::Full,
            Some(prev) => {
                if prev.message.message_stream_id != message.message_stream_id {
                    ChunkHeaderType::Full
                } else if prev.message.message_length != message.message_length
                    || prev.message.message_type != message.message_type
                {
                    ChunkHeaderType::SameStream
                } else if prev.message.timestamp != message.timestamp {
                    ChunkHeaderType::TimestampOnly
                } else {
                    ChunkHeaderType::Continuation
                }
            }
        };

        let header = ChunkHeader::new(format, chunk_stream_id, *message);

        // Write first chunk with header
        header.encode(&mut buf);

        let chunk_size = self.tx_chunk_size as usize;
        let first_chunk_size = chunk_size.min(payload.len());
        buf.put_slice(&payload[..first_chunk_size]);

        // Write continuation chunks
        let mut offset = first_chunk_size;
        while offset < payload.len() {
            let cont_header =
                ChunkHeader::new(ChunkHeaderType::Continuation, chunk_stream_id, *message);
            cont_header.encode_basic_header(&mut buf);

            // Extended timestamp for continuation if needed
            if message.needs_extended_timestamp() {
                buf.put_u32(message.timestamp);
            }

            let chunk_end = (offset + chunk_size).min(payload.len());
            buf.put_slice(&payload[offset..chunk_end]);
            offset = chunk_end;
        }

        // Save header
        self.prev_headers.insert(chunk_stream_id, header);

        buf.freeze()
    }
}

impl Default for ChunkStream {
    fn default() -> Self {
        Self::new()
    }
}

/// AMF0 value types for metadata encoding.
#[derive(Debug, Clone, PartialEq)]
pub enum Amf0Value {
    /// Number (f64).
    Number(f64),
    /// Boolean.
    Boolean(bool),
    /// String.
    String(String),
    /// Object (ordered key-value pairs).
    Object(Vec<(String, Amf0Value)>),
    /// Null.
    Null,
    /// ECMA Array.
    Array(Vec<(String, Amf0Value)>),
}

impl Amf0Value {
    /// Encodes this value into the given buffer.
    pub fn encode_into(&self, buf: &mut BytesMut) {
        match self {
            Self::Number(n) => {
                buf.put_u8(0x00); // Number marker
                buf.put_f64(*n);
            }
            Self::Boolean(b) => {
                buf.put_u8(0x01); // Boolean marker
                buf.put_u8(u8::from(*b));
            }
            Self::String(s) => {
                let bytes = s.as_bytes();
                if bytes.len() <= 0xFFFF {
                    buf.put_u8(0x02); // String marker
                    buf.put_u16(bytes.len() as u16);
                } else {
                    buf.put_u8(0x0C); // LongString marker
                    buf.put_u32(bytes.len() as u32);
                }
                buf.put_slice(bytes);
            }
            Self::Object(props) => {
                buf.put_u8(0x03); // Object marker
                for (key, val) in props {
                    let key_bytes = key.as_bytes();
                    buf.put_u16(key_bytes.len() as u16);
                    buf.put_slice(key_bytes);
                    val.encode_into(buf);
                }
                // Object end: empty key + ObjectEnd marker
                buf.put_u16(0);
                buf.put_u8(0x09);
            }
            Self::Null => {
                buf.put_u8(0x05); // Null marker
            }
            Self::Array(props) => {
                buf.put_u8(0x08); // ECMA Array marker
                buf.put_u32(props.len() as u32);
                for (key, val) in props {
                    let key_bytes = key.as_bytes();
                    buf.put_u16(key_bytes.len() as u16);
                    buf.put_slice(key_bytes);
                    val.encode_into(buf);
                }
                buf.put_u16(0);
                buf.put_u8(0x09);
            }
        }
    }

    /// Encodes this value and returns it as bytes.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();
        self.encode_into(&mut buf);
        buf.freeze()
    }
}

/// Stateful chunk decoder that reassembles RTMP messages from a byte stream.
///
/// Unlike `ChunkStream::process_chunk()` which requires complete chunk data,
/// `ChunkDecoder` handles partial data and maintains an internal buffer.
#[derive(Debug)]
pub struct ChunkDecoder {
    /// Internal buffer for incoming data.
    buffer: BytesMut,
    /// Chunk stream states.
    streams: HashMap<u32, ChunkStreamState>,
    /// Previous headers for delta decoding.
    prev_headers: HashMap<u32, ChunkHeader>,
    /// Current receive chunk size.
    chunk_size: u32,
}

impl ChunkDecoder {
    /// Creates a new chunk decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::new(),
            streams: HashMap::new(),
            prev_headers: HashMap::new(),
            chunk_size: DEFAULT_CHUNK_SIZE,
        }
    }

    /// Sets the receive chunk size (used after `SetChunkSize` negotiation).
    pub fn set_chunk_size(&mut self, size: u32) {
        self.chunk_size = size.min(MAX_CHUNK_SIZE);
    }

    /// Returns the current chunk size.
    #[must_use]
    pub const fn chunk_size(&self) -> u32 {
        self.chunk_size
    }

    /// Feeds raw bytes into the decoder and returns any complete messages.
    ///
    /// This method handles partial data: call it repeatedly as bytes arrive.
    /// Handles chunk type 0 (full/absolute), 1 (same stream, relative),
    /// 2 (timestamp only), and 3 (continuation/no header).
    ///
    /// Also processes `SetChunkSize` control messages automatically.
    ///
    /// # Errors
    ///
    /// Returns an error if the incoming data contains a malformed chunk.
    pub fn decode(&mut self, data: &[u8]) -> NetResult<Vec<AssembledMessage>> {
        self.buffer.put_slice(data);
        let mut messages = Vec::new();

        loop {
            let snapshot = self.buffer.len();
            match self.try_decode_one() {
                Ok(Some(msg)) => {
                    // Auto-handle SetChunkSize (message type 1)
                    if msg.header.message_type == 1 && msg.payload.len() >= 4 {
                        let p = msg.payload.as_ref();
                        let new_size = (u32::from(p[0]) << 24
                            | u32::from(p[1]) << 16
                            | u32::from(p[2]) << 8
                            | u32::from(p[3]))
                            & 0x7FFF_FFFF;
                        if new_size > 0 {
                            self.chunk_size = new_size.min(MAX_CHUNK_SIZE);
                        }
                    }
                    messages.push(msg);
                }
                Ok(None) => {
                    // No complete message yet. Stop only if no progress was made
                    // (i.e. the buffer didn't shrink -- we need more bytes from the network).
                    if self.buffer.len() >= snapshot {
                        break;
                    }
                    // Otherwise, bytes were consumed for partial assembly; keep going.
                }
                Err(e) => return Err(e),
            }
            // Guard against infinite loops when no bytes are consumed.
            if self.buffer.len() == snapshot {
                break;
            }
        }

        Ok(messages)
    }

    /// Attempts to decode a single complete message from the internal buffer.
    fn try_decode_one(&mut self) -> NetResult<Option<AssembledMessage>> {
        if self.buffer.is_empty() {
            return Ok(None);
        }

        // Peek at the buffer to check if we have a complete chunk header
        let buf_ref: &[u8] = &self.buffer;
        if buf_ref.is_empty() {
            return Ok(None);
        }

        let first = buf_ref[0];
        let fmt = first >> 6;
        let csid_marker = first & 0x3F;

        // Calculate basic header size
        let basic_header_size: usize = match csid_marker {
            0 => 2,
            1 => 3,
            _ => 1,
        };

        // Calculate message header size based on format
        let msg_header_size: usize = match fmt {
            0 => 11,
            1 => 7,
            2 => 3,
            _ => 0,
        };

        // Calculate minimum bytes needed for header (excluding extended timestamp)
        let min_needed = basic_header_size + msg_header_size;
        if self.buffer.len() < min_needed {
            return Ok(None);
        }

        // Peek at timestamp field to determine if extended timestamp is needed
        let needs_ext_ts = if msg_header_size >= 3 {
            let ts_start = basic_header_size;
            let ts = (u32::from(buf_ref[ts_start]) << 16)
                | (u32::from(buf_ref[ts_start + 1]) << 8)
                | u32::from(buf_ref[ts_start + 2]);
            ts == 0x00FF_FFFF
        } else {
            // For type 3, check previous header
            let csid = self.compute_csid(buf_ref);
            self.prev_headers
                .get(&csid)
                .and_then(|h| h.extended_timestamp)
                .is_some()
        };

        let ext_ts_size = if needs_ext_ts { 4 } else { 0 };
        let total_header_size = min_needed + ext_ts_size;

        if self.buffer.len() < total_header_size {
            return Ok(None);
        }

        // Decode header
        let header_bytes = self.buffer.clone().freeze();
        let mut slice: &[u8] = &header_bytes;
        let header = ChunkHeader::decode(&mut slice, &self.prev_headers)?;
        let consumed_header = total_header_size;

        let csid = header.chunk_stream_id;
        let state = self
            .streams
            .entry(csid)
            .or_insert_with(ChunkStreamState::new);

        // Start a new message if needed
        if !matches!(header.format, ChunkHeaderType::Continuation) || state.message_length == 0 {
            state.reset();
            state.message_length = header.message.message_length;
        }

        let remaining = state.message_length.saturating_sub(state.bytes_received);
        let chunk_payload_size = remaining.min(self.chunk_size) as usize;
        let total_needed = consumed_header + chunk_payload_size;

        if self.buffer.len() < total_needed {
            return Ok(None);
        }

        // Consume the data
        let _ = self.buffer.split_to(consumed_header);
        let payload_bytes = self.buffer.split_to(chunk_payload_size);

        let state = self
            .streams
            .entry(csid)
            .or_insert_with(ChunkStreamState::new);
        state.message_buffer.put_slice(&payload_bytes);
        state.bytes_received += chunk_payload_size as u32;

        let result = if state.bytes_received >= state.message_length {
            let assembled = AssembledMessage {
                chunk_stream_id: csid,
                header: header.message,
                payload: state.message_buffer.split().freeze(),
            };
            state.reset();
            Some(assembled)
        } else {
            None
        };

        self.prev_headers.insert(csid, header);
        Ok(result)
    }

    /// Computes the chunk stream ID from the first few bytes without advancing position.
    fn compute_csid(&self, buf: &[u8]) -> u32 {
        if buf.is_empty() {
            return 0;
        }
        let marker = buf[0] & 0x3F;
        match marker {
            0 => {
                if buf.len() >= 2 {
                    64 + u32::from(buf[1])
                } else {
                    0
                }
            }
            1 => {
                if buf.len() >= 3 {
                    64 + u32::from(buf[1]) + (u32::from(buf[2]) << 8)
                } else {
                    0
                }
            }
            _ => u32::from(marker),
        }
    }
}

impl Default for ChunkDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Stateful chunk encoder that splits RTMP messages into chunks.
///
/// Tracks per-stream state to use the most compact header type possible.
#[derive(Debug)]
pub struct ChunkEncoder {
    /// Transmit chunk size.
    chunk_size: u32,
    /// Previous headers per chunk stream ID (for delta encoding).
    prev_headers: HashMap<u32, ChunkHeader>,
}

impl ChunkEncoder {
    /// Creates a new chunk encoder with default chunk size.
    #[must_use]
    pub fn new() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            prev_headers: HashMap::new(),
        }
    }

    /// Creates a chunk encoder with a specific chunk size.
    #[must_use]
    pub fn with_chunk_size(chunk_size: u32) -> Self {
        Self {
            chunk_size: chunk_size.min(MAX_CHUNK_SIZE),
            prev_headers: HashMap::new(),
        }
    }

    /// Returns the current chunk size.
    #[must_use]
    pub const fn chunk_size(&self) -> u32 {
        self.chunk_size
    }

    /// Sets the chunk size.
    pub fn set_chunk_size(&mut self, size: u32) {
        self.chunk_size = size.min(MAX_CHUNK_SIZE);
    }

    /// Encodes a message header + payload into one or more chunks.
    ///
    /// Automatically selects the most compact chunk header type based on
    /// what has been sent previously on the given chunk stream.
    ///
    /// # Arguments
    ///
    /// * `chunk_stream_id` - The chunk stream ID (2-65599).
    /// * `message` - The message header describing this message.
    /// * `payload` - The raw message payload.
    ///
    /// # Returns
    ///
    /// The encoded chunk bytes ready to be sent on the wire.
    pub fn encode(
        &mut self,
        chunk_stream_id: u32,
        message: &MessageHeader,
        payload: &[u8],
    ) -> Bytes {
        let mut buf = BytesMut::new();

        let format = match self.prev_headers.get(&chunk_stream_id) {
            None => ChunkHeaderType::Full,
            Some(prev) => {
                if prev.message.message_stream_id != message.message_stream_id {
                    ChunkHeaderType::Full
                } else if prev.message.message_length != message.message_length
                    || prev.message.message_type != message.message_type
                {
                    ChunkHeaderType::SameStream
                } else if prev.message.timestamp != message.timestamp {
                    ChunkHeaderType::TimestampOnly
                } else {
                    ChunkHeaderType::Continuation
                }
            }
        };

        let header = ChunkHeader::new(format, chunk_stream_id, *message);
        header.encode(&mut buf);

        let chunk_size = self.chunk_size as usize;
        let first_end = chunk_size.min(payload.len());
        buf.put_slice(&payload[..first_end]);

        let mut offset = first_end;
        while offset < payload.len() {
            let cont = ChunkHeader::new(ChunkHeaderType::Continuation, chunk_stream_id, *message);
            cont.encode_basic_header(&mut buf);

            if message.needs_extended_timestamp() {
                buf.put_u32(message.timestamp);
            }

            let end = (offset + chunk_size).min(payload.len());
            buf.put_slice(&payload[offset..end]);
            offset = end;
        }

        self.prev_headers.insert(chunk_stream_id, header);
        buf.freeze()
    }

    /// Encodes a `SetChunkSize` control message.
    ///
    /// Also updates the internal chunk size.
    pub fn encode_set_chunk_size(&mut self, new_size: u32) -> Bytes {
        self.chunk_size = new_size.min(MAX_CHUNK_SIZE);
        let payload_val = new_size & 0x7FFF_FFFF;
        let mut payload = BytesMut::with_capacity(4);
        payload.put_u32(payload_val);

        let msg = MessageHeader::new(0, 4, 1, 0);
        // SetChunkSize always uses chunk stream 2 (protocol control)
        self.encode(2, &msg, &payload)
    }
}

impl Default for ChunkEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl ChunkStream {
    /// Test-only: current capacity of the reassembly buffer for `csid`, used to
    /// assert that a hostile `message_length` is not pre-reserved wholesale.
    fn debug_stream_capacity(&self, csid: u32) -> Option<usize> {
        self.streams.get(&csid).map(|s| s.message_buffer.capacity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_header_type() {
        assert_eq!(ChunkHeaderType::Full.format_value(), 0);
        assert_eq!(ChunkHeaderType::SameStream.format_value(), 1);
        assert_eq!(ChunkHeaderType::TimestampOnly.format_value(), 2);
        assert_eq!(ChunkHeaderType::Continuation.format_value(), 3);

        assert_eq!(ChunkHeaderType::from_format(0), Some(ChunkHeaderType::Full));
        assert_eq!(ChunkHeaderType::from_format(4), None);
    }

    #[test]
    fn test_message_header() {
        let msg = MessageHeader::new(1000, 500, 20, 1);
        assert_eq!(msg.timestamp, 1000);
        assert_eq!(msg.message_length, 500);
        assert!(!msg.needs_extended_timestamp());

        let msg2 = MessageHeader::new(0x00FF_FFFF, 100, 20, 1);
        assert!(msg2.needs_extended_timestamp());
    }

    #[test]
    fn test_chunk_header_encode_decode() {
        let msg = MessageHeader::new(1000, 100, 20, 1);
        let header = ChunkHeader::new(ChunkHeaderType::Full, 3, msg);

        let mut buf = BytesMut::new();
        header.encode(&mut buf);

        let encoded = buf.freeze();
        let mut slice = &encoded[..];
        let prev = HashMap::new();

        let decoded = ChunkHeader::decode(&mut slice, &prev).expect("should succeed in test");
        assert_eq!(decoded.chunk_stream_id, 3);
        assert_eq!(decoded.message.timestamp, 1000);
        assert_eq!(decoded.message.message_length, 100);
        assert_eq!(decoded.message.message_type, 20);
    }

    #[test]
    fn test_chunk_header_large_csid() {
        // Test 2-byte CSID
        let msg = MessageHeader::new(0, 100, 1, 0);
        let header = ChunkHeader::new(ChunkHeaderType::Full, 100, msg);

        let mut buf = BytesMut::new();
        header.encode(&mut buf);

        let encoded = buf.freeze();
        let mut slice = &encoded[..];
        let prev = HashMap::new();

        let decoded = ChunkHeader::decode(&mut slice, &prev).expect("should succeed in test");
        assert_eq!(decoded.chunk_stream_id, 100);
    }

    #[test]
    fn test_chunk_stream_encode_decode() {
        let mut cs = ChunkStream::new();
        cs.set_tx_chunk_size(128);
        cs.set_rx_chunk_size(128);

        let msg = MessageHeader::new(0, 50, 20, 1);
        let payload = vec![1u8; 50];

        let encoded = cs.encode_message(3, &msg, &payload);

        let messages = cs.process_chunk(&encoded).expect("should succeed in test");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].payload.len(), 50);
    }

    #[test]
    fn huge_declared_message_length_does_not_over_reserve() {
        // A Type-0 header can declare up to a 16 MiB message. Feeding one such
        // header with only a single 128-byte chunk of payload must not cause
        // the per-stream buffer to reserve the whole declared length (the ~1 TB
        // amplification this guard defends). The message stays incomplete.
        let mut cs = ChunkStream::new();
        cs.set_rx_chunk_size(128);

        let msg = MessageHeader::new(0, 0x00FF_FFFF, 9, 1);
        let header = ChunkHeader::new(ChunkHeaderType::Full, 3, msg);
        let mut buf = BytesMut::new();
        header.encode(&mut buf);
        buf.put_slice(&[0xAB; 128]);

        let messages = cs
            .process_chunk(&buf)
            .expect("partial chunk should buffer cleanly");
        assert!(messages.is_empty(), "incomplete message must not assemble");

        let cap = cs
            .debug_stream_capacity(3)
            .expect("stream state should exist");
        assert!(
            cap < 1024 * 1024,
            "reassembly buffer over-reserved from declared length: cap={cap}"
        );
    }

    #[test]
    fn test_chunk_stream_large_message() {
        let mut cs = ChunkStream::new();
        cs.set_tx_chunk_size(128);
        cs.set_rx_chunk_size(128);

        // Message larger than chunk size
        let msg = MessageHeader::new(0, 300, 20, 1);
        let payload = vec![0xAB; 300];

        let encoded = cs.encode_message(3, &msg, &payload);

        let messages = cs.process_chunk(&encoded).expect("should succeed in test");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].payload.len(), 300);
        assert_eq!(&messages[0].payload[..], &payload[..]);
    }

    #[test]
    fn test_extended_timestamp() {
        let msg = MessageHeader::new(0x01000000, 10, 1, 0);
        let header = ChunkHeader::new(ChunkHeaderType::Full, 3, msg);

        assert!(header.extended_timestamp.is_some());
        assert_eq!(header.timestamp(), 0x01000000);
    }

    // --- ChunkDecoder tests ---

    #[test]
    fn test_chunk_decoder_simple_message() {
        let mut encoder = ChunkEncoder::with_chunk_size(128);
        let msg = MessageHeader::new(100, 32, 20, 1);
        let payload = vec![0xBB; 32];
        let encoded = encoder.encode(3, &msg, &payload);

        let mut decoder = ChunkDecoder::new();
        let messages = decoder.decode(&encoded).expect("should succeed in test");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].payload.len(), 32);
        assert_eq!(&messages[0].payload[..], &payload[..]);
    }

    #[test]
    fn test_chunk_decoder_large_message() {
        let mut encoder = ChunkEncoder::with_chunk_size(128);
        let msg = MessageHeader::new(0, 512, 9, 1);
        let payload = (0u8..=255u8).cycle().take(512).collect::<Vec<_>>();
        let encoded = encoder.encode(4, &msg, &payload);

        let mut decoder = ChunkDecoder::new();
        let messages = decoder.decode(&encoded).expect("should succeed in test");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].payload.len(), 512);
        assert_eq!(&messages[0].payload[..], &payload[..]);
    }

    #[test]
    fn test_chunk_decoder_incremental_feed() {
        // Encoder and decoder must agree on chunk_size (this is negotiated in real RTMP).
        let mut encoder = ChunkEncoder::with_chunk_size(64);
        let msg = MessageHeader::new(500, 128, 8, 1);
        let payload = vec![0xCC; 128];
        let encoded = encoder.encode(3, &msg, &payload);

        // Feed in small increments - decoder uses same chunk size as encoder
        let mut decoder = ChunkDecoder::new();
        decoder.set_chunk_size(64);
        let mut all_messages = Vec::new();
        for chunk in encoded.chunks(10) {
            let msgs = decoder.decode(chunk).expect("should succeed in test");
            all_messages.extend(msgs);
        }
        assert_eq!(all_messages.len(), 1);
        assert_eq!(&all_messages[0].payload[..], &payload[..]);
    }

    #[test]
    fn test_chunk_decoder_multiple_messages() {
        let mut encoder = ChunkEncoder::with_chunk_size(128);
        let msg1 = MessageHeader::new(0, 20, 8, 1);
        let payload1 = vec![0x11; 20];
        let msg2 = MessageHeader::new(100, 20, 9, 1);
        let payload2 = vec![0x22; 20];

        let mut combined = Vec::new();
        combined.extend_from_slice(&encoder.encode(3, &msg1, &payload1));
        combined.extend_from_slice(&encoder.encode(4, &msg2, &payload2));

        let mut decoder = ChunkDecoder::new();
        let messages = decoder.decode(&combined).expect("should succeed in test");
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_chunk_decoder_set_chunk_size() {
        let mut decoder = ChunkDecoder::new();
        assert_eq!(decoder.chunk_size(), DEFAULT_CHUNK_SIZE);
        decoder.set_chunk_size(4096);
        assert_eq!(decoder.chunk_size(), 4096);
        decoder.set_chunk_size(MAX_CHUNK_SIZE + 1000);
        assert_eq!(decoder.chunk_size(), MAX_CHUNK_SIZE);
    }

    #[test]
    fn test_chunk_decoder_empty_input() {
        let mut decoder = ChunkDecoder::new();
        let messages = decoder.decode(&[]).expect("should succeed in test");
        assert!(messages.is_empty());
    }

    // --- ChunkEncoder tests ---

    #[test]
    fn test_chunk_encoder_default_chunk_size() {
        let enc = ChunkEncoder::new();
        assert_eq!(enc.chunk_size(), DEFAULT_CHUNK_SIZE);
    }

    #[test]
    fn test_chunk_encoder_with_chunk_size() {
        let enc = ChunkEncoder::with_chunk_size(4096);
        assert_eq!(enc.chunk_size(), 4096);
    }

    #[test]
    fn test_chunk_encoder_set_chunk_size_capped() {
        let mut enc = ChunkEncoder::new();
        enc.set_chunk_size(MAX_CHUNK_SIZE + 1);
        assert_eq!(enc.chunk_size(), MAX_CHUNK_SIZE);
    }

    #[test]
    fn test_chunk_encoder_encode_small_payload() {
        let mut enc = ChunkEncoder::with_chunk_size(256);
        let msg = MessageHeader::new(0, 10, 20, 1);
        let payload = vec![0xAA; 10];
        let encoded = enc.encode(3, &msg, &payload);

        // Decode with ChunkDecoder and verify
        let mut dec = ChunkDecoder::new();
        dec.set_chunk_size(256);
        let messages = dec.decode(&encoded).expect("should succeed in test");
        assert_eq!(messages.len(), 1);
        assert_eq!(&messages[0].payload[..], &payload[..]);
    }

    #[test]
    fn test_chunk_encoder_header_compression() {
        // First message: Full header. Second message with same stream and length
        // but different timestamp: TimestampOnly header.
        let mut enc = ChunkEncoder::with_chunk_size(512);
        let msg1 = MessageHeader::new(0, 10, 8, 1);
        let msg2 = MessageHeader::new(33, 10, 8, 1);
        let payload = vec![0u8; 10];

        let enc1 = enc.encode(3, &msg1, &payload);
        let enc2 = enc.encode(3, &msg2, &payload);

        // enc2 should be shorter since it uses a compressed header
        assert!(enc2.len() < enc1.len());
    }

    #[test]
    fn test_chunk_encoder_encode_set_chunk_size() {
        let mut enc = ChunkEncoder::new();
        let encoded = enc.encode_set_chunk_size(4096);
        assert!(!encoded.is_empty());
        assert_eq!(enc.chunk_size(), 4096);
    }

    // --- Amf0Value tests ---

    #[test]
    fn test_amf0_number_encoding() {
        let val = Amf0Value::Number(42.5);
        let encoded = val.encode();
        assert_eq!(encoded[0], 0x00); // Number marker
        assert_eq!(encoded.len(), 9); // 1 marker + 8 bytes f64
    }

    #[test]
    fn test_amf0_boolean_encoding() {
        let val_true = Amf0Value::Boolean(true);
        let encoded_true = val_true.encode();
        assert_eq!(encoded_true[0], 0x01);
        assert_eq!(encoded_true[1], 1);

        let val_false = Amf0Value::Boolean(false);
        let encoded_false = val_false.encode();
        assert_eq!(encoded_false[1], 0);
    }

    #[test]
    fn test_amf0_string_encoding() {
        let val = Amf0Value::String("hello".to_string());
        let encoded = val.encode();
        assert_eq!(encoded[0], 0x02); // String marker
        assert_eq!(encoded[1], 0); // high byte of length
        assert_eq!(encoded[2], 5); // low byte of length (5 chars)
        assert_eq!(&encoded[3..], b"hello");
    }

    #[test]
    fn test_amf0_null_encoding() {
        let val = Amf0Value::Null;
        let encoded = val.encode();
        assert_eq!(encoded[0], 0x05);
        assert_eq!(encoded.len(), 1);
    }

    #[test]
    fn test_amf0_object_encoding() {
        let props = vec![
            ("width".to_string(), Amf0Value::Number(1920.0)),
            ("height".to_string(), Amf0Value::Number(1080.0)),
        ];
        let val = Amf0Value::Object(props);
        let encoded = val.encode();
        assert_eq!(encoded[0], 0x03); // Object marker
                                      // Should end with 0x00 0x00 0x09 (empty key + ObjectEnd)
        assert_eq!(*encoded.last().expect("should succeed in test"), 0x09);
    }

    #[test]
    fn test_amf0_array_encoding() {
        let props = vec![("fps".to_string(), Amf0Value::Number(30.0))];
        let val = Amf0Value::Array(props);
        let encoded = val.encode();
        assert_eq!(encoded[0], 0x08); // ECMA Array marker
                                      // bytes 1-4 are the count (u32 big-endian): count = 1
        assert_eq!(encoded[1], 0); // highest byte of count
        assert_eq!(encoded[2], 0);
        assert_eq!(encoded[3], 0);
        assert_eq!(encoded[4], 1); // lowest byte: 1 element
    }

    #[test]
    fn test_amf0_encode_into_buf() {
        let mut buf = BytesMut::new();
        let num = Amf0Value::Number(1.0);
        let s = Amf0Value::String("test".to_string());
        num.encode_into(&mut buf);
        s.encode_into(&mut buf);
        // 9 bytes for number + (1+2+4) = 7 bytes for string = 16 total
        assert_eq!(buf.len(), 9 + 7);
    }

    // --- Byte-level header layout reference tests ---

    #[test]
    fn test_chunk_type0_header_byte_layout() {
        // timestamp=0x001234, length=100=0x64, type=20=0x14, stream_id=1
        let msg = MessageHeader::new(0x001234, 0x000064, 0x14, 0x00000001);
        // CSID=3 fits in 1-byte basic header
        let header = ChunkHeader::new(ChunkHeaderType::Full, 3, msg);

        let mut buf = BytesMut::new();
        header.encode(&mut buf);
        let encoded = buf.freeze();

        // Byte 0: basic header = (fmt=0 << 6) | csid=3 = 0x03
        assert_eq!(
            encoded[0], 0x03,
            "basic header byte must be (0<<6)|3 = 0x03"
        );

        // Bytes 1..4: timestamp big-endian 3 bytes = 0x001234 → [0x00, 0x12, 0x34]
        assert_eq!(
            &encoded[1..4],
            &[0x00, 0x12, 0x34],
            "timestamp must be big-endian 0x001234"
        );

        // Bytes 4..7: message_length big-endian 3 bytes = 100 = 0x000064 → [0x00, 0x00, 0x64]
        assert_eq!(
            &encoded[4..7],
            &[0x00, 0x00, 0x64],
            "message_length must be big-endian 100"
        );

        // Byte 7: message_type = 20 = 0x14
        assert_eq!(encoded[7], 0x14, "message_type must be 0x14 (20)");

        // Bytes 8..12: message_stream_id little-endian = 1 → [0x01, 0x00, 0x00, 0x00]
        assert_eq!(
            &encoded[8..12],
            &[0x01, 0x00, 0x00, 0x00],
            "stream_id must be little-endian 1"
        );
    }

    #[test]
    fn test_chunk_type1_header_byte_layout() {
        let csid = 4u32;
        let mut encoder = ChunkEncoder::with_chunk_size(512);

        // First message: Full (type 0) header establishes stream state
        let msg1 = MessageHeader::new(0, 10, 8, 1);
        let payload1 = vec![0u8; 10];
        let _ = encoder.encode(csid, &msg1, &payload1);

        // Second message: same stream_id=1, different length=20 → SameStream (type 1) header
        let msg2 = MessageHeader::new(100, 20, 8, 1);
        let payload2 = vec![0u8; 20];
        let encoded2 = encoder.encode(csid, &msg2, &payload2);

        // Basic header byte 0 for SameStream (fmt=1) with csid=4 in 1-byte form:
        // (1 << 6) | 4 = 0x44
        let expected_basic = (1u8 << 6) | (csid as u8);
        assert_eq!(
            encoded2[0], expected_basic,
            "SameStream basic header must have fmt=1 in top 2 bits"
        );
        assert_eq!(encoded2[0], 0x44, "basic header byte must be 0x44");
    }

    #[test]
    fn test_chunk_full_encode_decode_reference() {
        // Build a 20-byte alternating payload (reference byte vector)
        let payload: Vec<u8> = (0..20)
            .map(|i| if i % 2 == 0 { 0xDE } else { 0xAD })
            .collect();

        let mut encoder = ChunkEncoder::with_chunk_size(128);
        let msg = MessageHeader::new(0, 20, 20, 1);
        let encoded = encoder.encode(3, &msg, &payload);

        let mut decoder = ChunkDecoder::new();
        let messages = decoder
            .decode(&encoded)
            .expect("decode must succeed on valid chunk data");

        assert_eq!(messages.len(), 1, "must decode exactly one message");
        assert_eq!(
            messages[0].payload.len(),
            payload.len(),
            "decoded payload length must match original"
        );
        assert_eq!(
            &messages[0].payload[..],
            &payload[..],
            "decoded payload must be byte-exact match to original"
        );
    }
}
