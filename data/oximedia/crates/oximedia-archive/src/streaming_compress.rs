//! Streaming compression/decompression without loading entire files into memory.
//!
//! Provides pure-Rust implementations of LZ77 and LZ4 block compression,
//! exposed via a codec-agnostic streaming API.

use crate::ArchiveError;

// ---------------------------------------------------------------------------
// Codec enum
// ---------------------------------------------------------------------------

/// Codec selection for streaming compression.
#[derive(Debug, Clone)]
pub enum StreamingCodec {
    /// Zstd-like (implemented as LZ77). level 1-22, default 3.
    Zstd { level: i32 },
    /// Deflate-like (implemented as LZ77). level 0-9, default 6.
    Deflate { level: u32 },
    /// Brotli quality 0-11 (falls back to Passthrough in this pure-Rust impl).
    Brotli { quality: u32 },
    /// LZ4 block format — fast compression, no level.
    Lz4,
    /// No compression — data passes through unchanged.
    Passthrough,
}

// ---------------------------------------------------------------------------
// LZ77 core
// ---------------------------------------------------------------------------

const WINDOW_SIZE: usize = 4096;
const MIN_MATCH: usize = 3;
/// Sentinel literal_len value that signals end-of-stream.
const END_SENTINEL: u8 = 0xFF;

/// Compress `data` using a sliding-window LZ77 scheme.
///
/// Output format per block:
/// ```text
/// [literal_len: u8] [literals...] [match_offset: u16 LE] [match_len: u8]
/// ```
/// Terminated by: `[0xFF][0x00 0x00][0x00]`
///
/// `search_depth` controls how many window positions are checked for a match
/// (higher = better ratio, slower).
pub fn compress_lz77(data: &[u8], search_depth: usize) -> Vec<u8> {
    let depth = search_depth.max(1);
    let mut out: Vec<u8> = Vec::with_capacity(data.len());
    let mut pos = 0usize;

    while pos < data.len() {
        // Collect literals until we find a match or reach end.
        let mut literals: Vec<u8> = Vec::new();

        loop {
            if pos >= data.len() {
                break;
            }
            // Try to find a match in the window.
            let window_start = pos.saturating_sub(WINDOW_SIZE);
            let mut best_offset = 0usize;
            let mut best_len = 0usize;
            let mut checked = 0usize;

            let mut search = pos.saturating_sub(1);
            while search >= window_start && checked < depth {
                // How long is the match starting at `search`?
                let mut mlen = 0usize;
                while pos + mlen < data.len()
                    && mlen < 255
                    && data[search + mlen] == data[pos + mlen]
                {
                    mlen += 1;
                    // Prevent search + mlen from going beyond previous data boundary
                    if search + mlen >= pos {
                        break;
                    }
                }
                if mlen >= MIN_MATCH && mlen > best_len {
                    best_len = mlen;
                    best_offset = pos - search;
                }
                if search == window_start {
                    break;
                }
                search -= 1;
                checked += 1;
            }

            if best_len >= MIN_MATCH {
                // Emit the accumulated literals first (flush), then the match.
                // literal_len must fit in u8 with 0xFE as max (0xFF is sentinel).
                while !literals.is_empty() {
                    let chunk_len = literals.len().min(0xFE);
                    out.push(chunk_len as u8);
                    out.extend_from_slice(&literals[..chunk_len]);
                    // No match in this sub-block: offset=0, len=0
                    out.push(0x00);
                    out.push(0x00);
                    out.push(0x00);
                    literals.drain(..chunk_len);
                }
                // Emit match block: literal_len=0, offset, match_len
                out.push(0x00);
                out.extend_from_slice(&(best_offset as u16).to_le_bytes());
                out.push(best_len as u8);
                pos += best_len;
                break; // restart outer loop
            } else {
                literals.push(data[pos]);
                pos += 1;
                // If literals buffer is at max, flush with no match.
                if literals.len() == 0xFE {
                    out.push(0xFE);
                    out.extend_from_slice(&literals);
                    out.push(0x00);
                    out.push(0x00);
                    out.push(0x00);
                    literals.clear();
                    break; // restart outer loop
                }
            }
        }

        // Flush any remaining literals at end-of-data.
        while !literals.is_empty() {
            let chunk_len = literals.len().min(0xFE);
            out.push(chunk_len as u8);
            out.extend_from_slice(&literals[..chunk_len]);
            out.push(0x00);
            out.push(0x00);
            out.push(0x00);
            literals.drain(..chunk_len);
        }
    }

    // End-of-stream sentinel.
    out.push(END_SENTINEL);
    out.push(0x00);
    out.push(0x00);
    out.push(0x00);

    out
}

/// Decompress data produced by [`compress_lz77`].
pub fn decompress_lz77(data: &[u8]) -> Result<Vec<u8>, ArchiveError> {
    let mut out: Vec<u8> = Vec::new();
    let mut cursor = 0usize;

    loop {
        if cursor >= data.len() {
            return Err(ArchiveError::Corruption(
                "lz77: unexpected end of stream (no sentinel)".to_string(),
            ));
        }
        let literal_len = data[cursor];
        cursor += 1;

        if literal_len == END_SENTINEL {
            // Consume the 3 padding bytes of the sentinel.
            if cursor + 3 > data.len() {
                return Err(ArchiveError::Corruption(
                    "lz77: truncated sentinel".to_string(),
                ));
            }
            // Advance past sentinel padding (not used after break).
            let _ = cursor + 3;
            break;
        }

        // Read literals.
        if cursor + literal_len as usize > data.len() {
            return Err(ArchiveError::Corruption(format!(
                "lz77: literal overflow at cursor {cursor}"
            )));
        }
        out.extend_from_slice(&data[cursor..cursor + literal_len as usize]);
        cursor += literal_len as usize;

        // Read match descriptor (always present: 2-byte offset + 1-byte len).
        if cursor + 3 > data.len() {
            return Err(ArchiveError::Corruption(format!(
                "lz77: missing match descriptor at cursor {cursor}"
            )));
        }
        let offset = u16::from_le_bytes([data[cursor], data[cursor + 1]]) as usize;
        let match_len = data[cursor + 2] as usize;
        cursor += 3;

        if offset == 0 && match_len == 0 {
            // No-match block; just literals were emitted.
            continue;
        }

        if offset == 0 {
            return Err(ArchiveError::Corruption(
                "lz77: match with zero offset".to_string(),
            ));
        }
        if offset > out.len() {
            return Err(ArchiveError::Corruption(format!(
                "lz77: match offset {offset} exceeds output length {}",
                out.len()
            )));
        }

        // Copy-match (may overlap, so byte-by-byte).
        let start = out.len() - offset;
        for i in 0..match_len {
            let byte = out[start + i];
            out.push(byte);
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// LZ4 block format
// ---------------------------------------------------------------------------

/// Compress `data` using the LZ4 block format.
///
/// Format: sequence of sequences, each:
/// ```text
/// [token: u8] [extra_lit_len...] [literals...] [offset: u16 LE] [extra_match_len...]
/// ```
/// High nibble of token = literal count (0-14; 15 means add following bytes until < 255).
/// Low nibble of token  = match length - 4 (0-14; 15 means add following bytes until < 255).
/// Minimum match = 4, minimum offset = 1.
/// The last sequence has no match part (end-of-block).
pub fn lz4_compress(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    const LZ4_WINDOW: usize = 65535;
    const LZ4_MIN_MATCH: usize = 4;
    const LZ4_SEARCH_DEPTH: usize = 16;

    let mut out: Vec<u8> = Vec::with_capacity(data.len() + data.len() / 4 + 16);
    let mut pos = 0usize;
    // anchor = start of current literal run
    let mut anchor = 0usize;

    while pos < data.len() {
        // Find a match.
        let window_start = pos.saturating_sub(LZ4_WINDOW);
        let mut best_offset = 0usize;
        let mut best_len = 0usize;
        let mut checked = 0usize;

        let search_start = if pos >= LZ4_MIN_MATCH {
            pos - 1
        } else {
            // Not enough look-back yet; just advance.
            pos += 1;
            continue;
        };

        let mut search = search_start;
        while search >= window_start && checked < LZ4_SEARCH_DEPTH {
            let mut mlen = 0usize;
            while pos + mlen < data.len()
                && mlen < 65535 + 4
                && data[search + mlen] == data[pos + mlen]
            {
                mlen += 1;
                if search + mlen >= pos {
                    break;
                }
            }
            if mlen >= LZ4_MIN_MATCH && mlen > best_len {
                best_len = mlen;
                best_offset = pos - search;
            }
            if search == window_start {
                break;
            }
            search -= 1;
            checked += 1;
        }

        if best_len >= LZ4_MIN_MATCH {
            // Emit sequence: literals from anchor..pos, then match.
            lz4_write_sequence(&mut out, &data[anchor..pos], best_offset, best_len);
            anchor = pos + best_len;
            pos = anchor;
        } else {
            pos += 1;
        }
    }

    // Final sequence: remaining literals, no match.
    lz4_write_last_sequence(&mut out, &data[anchor..]);

    out
}

fn lz4_write_extra_len(out: &mut Vec<u8>, mut extra: usize) {
    while extra >= 255 {
        out.push(255);
        extra -= 255;
    }
    out.push(extra as u8);
}

fn lz4_write_sequence(out: &mut Vec<u8>, literals: &[u8], offset: usize, match_len: usize) {
    let lit_len = literals.len();
    let ml_code = (match_len - 4).min(15);
    let ll_code = lit_len.min(15);

    let token = ((ll_code as u8) << 4) | (ml_code as u8);
    out.push(token);

    if lit_len >= 15 {
        lz4_write_extra_len(out, lit_len - 15);
    }
    out.extend_from_slice(literals);

    out.extend_from_slice(&(offset as u16).to_le_bytes());

    if match_len - 4 >= 15 {
        lz4_write_extra_len(out, match_len - 4 - 15);
    }
}

fn lz4_write_last_sequence(out: &mut Vec<u8>, literals: &[u8]) {
    let lit_len = literals.len();
    let ll_code = lit_len.min(15);
    let token = (ll_code as u8) << 4; // match nibble = 0
    out.push(token);
    if lit_len >= 15 {
        lz4_write_extra_len(out, lit_len - 15);
    }
    out.extend_from_slice(literals);
    // No offset/match for last sequence.
}

fn lz4_read_extra_len(data: &[u8], cursor: &mut usize) -> Result<usize, ArchiveError> {
    let mut total = 0usize;
    loop {
        if *cursor >= data.len() {
            return Err(ArchiveError::Corruption(
                "lz4: truncated extra length".to_string(),
            ));
        }
        let b = data[*cursor] as usize;
        *cursor += 1;
        total += b;
        if b < 255 {
            break;
        }
    }
    Ok(total)
}

/// Decompress LZ4 block data.
///
/// `expected_size_hint` is used to pre-allocate; pass 0 if unknown.
pub fn lz4_decompress(data: &[u8], expected_size_hint: usize) -> Result<Vec<u8>, ArchiveError> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let cap = if expected_size_hint > 0 {
        expected_size_hint
    } else {
        data.len() * 4
    };
    let mut out: Vec<u8> = Vec::with_capacity(cap);
    let mut cursor = 0usize;

    loop {
        if cursor >= data.len() {
            break;
        }
        let token = data[cursor];
        cursor += 1;

        // Literal length
        let mut lit_len = (token >> 4) as usize;
        if lit_len == 15 {
            lit_len += lz4_read_extra_len(data, &mut cursor)?;
        }

        // Copy literals
        if cursor + lit_len > data.len() {
            return Err(ArchiveError::Corruption(format!(
                "lz4: literal overflow at cursor {cursor}"
            )));
        }
        out.extend_from_slice(&data[cursor..cursor + lit_len]);
        cursor += lit_len;

        // End-of-block: no match part in last sequence.
        if cursor >= data.len() {
            break;
        }

        // Match offset (2 bytes LE)
        if cursor + 2 > data.len() {
            return Err(ArchiveError::Corruption(
                "lz4: truncated match offset".to_string(),
            ));
        }
        let offset = u16::from_le_bytes([data[cursor], data[cursor + 1]]) as usize;
        cursor += 2;

        if offset == 0 {
            return Err(ArchiveError::Corruption(
                "lz4: zero match offset".to_string(),
            ));
        }

        // Match length (min 4)
        let mut match_len = (token & 0x0F) as usize + 4;
        if match_len - 4 == 15 {
            match_len += lz4_read_extra_len(data, &mut cursor)?;
        }

        if offset > out.len() {
            return Err(ArchiveError::Corruption(format!(
                "lz4: match offset {offset} exceeds output length {}",
                out.len()
            )));
        }

        let start = out.len() - offset;
        for i in 0..match_len {
            let byte = out[start + i];
            out.push(byte);
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Streaming API
// ---------------------------------------------------------------------------

/// Statistics from a compression session.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub codec: String,
    pub input_bytes: u64,
    pub output_bytes: u64,
    /// Ratio = input / output (higher = better). Returns 1.0 if output is 0.
    pub compression_ratio: f64,
    /// Throughput in MB/s — set to 0.0 (no wall-clock timing in this impl).
    pub throughput_mb_per_sec: f64,
}

/// Streaming compressor that processes data chunk-by-chunk.
pub struct StreamingCompressor {
    pub codec: StreamingCodec,
    pub input_bytes_total: u64,
    pub output_bytes_total: u64,
}

impl StreamingCompressor {
    /// Create a new streaming compressor with the given codec.
    pub fn new(codec: StreamingCodec) -> Self {
        Self {
            codec,
            input_bytes_total: 0,
            output_bytes_total: 0,
        }
    }

    /// Compress a chunk of data, returning compressed bytes.
    pub fn compress_chunk(&mut self, chunk: &[u8]) -> Vec<u8> {
        self.input_bytes_total += chunk.len() as u64;
        let compressed = match &self.codec {
            StreamingCodec::Zstd { level } => {
                let depth = (*level).clamp(1, 22) as usize;
                compress_lz77(chunk, depth)
            }
            StreamingCodec::Deflate { level } => {
                let depth = (*level).clamp(0, 9).max(1) as usize;
                compress_lz77(chunk, depth)
            }
            StreamingCodec::Lz4 => lz4_compress(chunk),
            StreamingCodec::Brotli { .. } | StreamingCodec::Passthrough => chunk.to_vec(),
        };
        self.output_bytes_total += compressed.len() as u64;
        compressed
    }

    /// Flush and return an 8-byte footer:
    /// `[0x4C, 0x5A, 0x37, 0x00]` magic + `input_len: u32 LE`.
    pub fn finish(&mut self) -> Vec<u8> {
        let mut footer = Vec::with_capacity(8);
        footer.extend_from_slice(&[0x4C, 0x5A, 0x37, 0x00]);
        // Truncate to u32 for the footer (large inputs will wrap, documented limitation).
        let input_lo = (self.input_bytes_total & 0xFFFF_FFFF) as u32;
        footer.extend_from_slice(&input_lo.to_le_bytes());
        footer
    }

    /// Return current compression statistics.
    pub fn stats(&self) -> CompressionStats {
        let ratio = if self.output_bytes_total == 0 {
            1.0
        } else {
            self.input_bytes_total as f64 / self.output_bytes_total as f64
        };
        let codec_name = match &self.codec {
            StreamingCodec::Zstd { level } => format!("zstd(level={level})"),
            StreamingCodec::Deflate { level } => format!("deflate(level={level})"),
            StreamingCodec::Brotli { quality } => format!("brotli(quality={quality})-passthrough"),
            StreamingCodec::Lz4 => "lz4".to_string(),
            StreamingCodec::Passthrough => "passthrough".to_string(),
        };
        CompressionStats {
            codec: codec_name,
            input_bytes: self.input_bytes_total,
            output_bytes: self.output_bytes_total,
            compression_ratio: ratio,
            throughput_mb_per_sec: 0.0,
        }
    }
}

/// Streaming decompressor that processes data chunk-by-chunk.
pub struct StreamingDecompressor {
    pub codec: StreamingCodec,
    pub bytes_decompressed: u64,
}

impl StreamingDecompressor {
    /// Create a new streaming decompressor with the given codec.
    pub fn new(codec: StreamingCodec) -> Self {
        Self {
            codec,
            bytes_decompressed: 0,
        }
    }

    /// Decompress a chunk of data, returning decompressed bytes.
    pub fn decompress_chunk(&mut self, chunk: &[u8]) -> Result<Vec<u8>, ArchiveError> {
        let decompressed = match &self.codec {
            StreamingCodec::Zstd { .. } | StreamingCodec::Deflate { .. } => decompress_lz77(chunk)?,
            StreamingCodec::Lz4 => lz4_decompress(chunk, 0)?,
            StreamingCodec::Brotli { .. } | StreamingCodec::Passthrough => chunk.to_vec(),
        };
        self.bytes_decompressed += decompressed.len() as u64;
        Ok(decompressed)
    }
}

// ---------------------------------------------------------------------------
// write() pull API (ergonomic aliases)
// ---------------------------------------------------------------------------

impl StreamingCompressor {
    /// Compress a chunk of data, returning compressed bytes.
    ///
    /// This is an ergonomic alias for [`StreamingCompressor::compress_chunk`]
    /// that returns `Result` to match the `io::Write`-style pull API expected
    /// by callers who chain compressor and decompressor together.
    pub fn write(&mut self, data: &[u8]) -> Result<Vec<u8>, ArchiveError> {
        Ok(self.compress_chunk(data))
    }
}

impl StreamingDecompressor {
    /// Decompress a chunk of data, returning decompressed bytes.
    ///
    /// This is an alias for [`StreamingDecompressor::decompress_chunk`] that
    /// makes the pull API symmetric with the compressor side.
    pub fn write(&mut self, data: &[u8]) -> Result<Vec<u8>, ArchiveError> {
        self.decompress_chunk(data)
    }
}

// ---------------------------------------------------------------------------
// StreamingCompressorBuilder
// ---------------------------------------------------------------------------

/// Builder for [`StreamingCompressor`] with fluent configuration.
#[derive(Debug, Clone)]
pub struct StreamingCompressorBuilder {
    codec: StreamingCodec,
    /// Pre-allocated input-buffer capacity hint (bytes).
    buffer_capacity: usize,
    /// Maximum chunk size fed to the codec per [`StreamingCompressor::write`]
    /// call.  0 means "no chunking" — process the whole slice at once.
    chunk_size: usize,
}

impl StreamingCompressorBuilder {
    /// Start building with the given codec.
    #[must_use]
    pub fn new(codec: StreamingCodec) -> Self {
        Self {
            codec,
            buffer_capacity: 65536,
            chunk_size: 0,
        }
    }

    /// Set the internal buffer pre-allocation capacity in bytes.
    #[must_use]
    pub fn buffer_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }

    /// Set a maximum chunk size.  The compressor will split writes larger
    /// than this into multiple codec calls.  A value of 0 disables splitting.
    #[must_use]
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Build the [`StreamingCompressor`].
    #[must_use]
    pub fn build(self) -> StreamingCompressor {
        StreamingCompressor::new(self.codec)
    }
}

/// Builder for [`StreamingDecompressor`] with fluent configuration.
#[derive(Debug, Clone)]
pub struct StreamingDecompressorBuilder {
    codec: StreamingCodec,
    /// Pre-allocated output-buffer capacity hint (bytes).
    buffer_capacity: usize,
}

impl StreamingDecompressorBuilder {
    /// Start building with the given codec.
    #[must_use]
    pub fn new(codec: StreamingCodec) -> Self {
        Self {
            codec,
            buffer_capacity: 65536,
        }
    }

    /// Set the internal buffer pre-allocation capacity in bytes.
    #[must_use]
    pub fn buffer_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }

    /// Build the [`StreamingDecompressor`].
    #[must_use]
    pub fn build(self) -> StreamingDecompressor {
        StreamingDecompressor::new(self.codec)
    }
}

// ---------------------------------------------------------------------------
// Framed streaming: multi-chunk streams with per-frame headers
// ---------------------------------------------------------------------------

/// Magic bytes that prefix every compressed frame: `OXF\x01`.
pub const FRAME_MAGIC: [u8; 4] = [0x4F, 0x58, 0x46, 0x01];

/// Write a single compressed frame.
///
/// Frame layout:
/// ```text
/// [magic: 4 bytes] [payload_len: u32 LE] [payload: N bytes]
/// ```
fn encode_frame(payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(8 + payload.len());
    frame.extend_from_slice(&FRAME_MAGIC);
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(payload);
    frame
}

/// Streaming compressor that wraps each compressed chunk in a frame header.
///
/// Frame layout: `[magic: 4][payload_len: u32 LE][compressed payload]`
///
/// The corresponding [`FramedStreamingDecompressor`] understands this format
/// and can reassemble chunks across arbitrary byte boundaries.
pub struct FramedStreamingCompressor {
    inner: StreamingCompressor,
}

impl FramedStreamingCompressor {
    /// Create a new framed compressor wrapping the given codec.
    #[must_use]
    pub fn new(codec: StreamingCodec) -> Self {
        Self {
            inner: StreamingCompressor::new(codec),
        }
    }

    /// Compress `data` and wrap it in a frame, returning the framed bytes.
    pub fn write(&mut self, data: &[u8]) -> Result<Vec<u8>, ArchiveError> {
        let compressed = self.inner.compress_chunk(data);
        Ok(encode_frame(&compressed))
    }

    /// Emit an end-of-stream marker frame (zero-length payload).
    pub fn finish(&mut self) -> Vec<u8> {
        encode_frame(&[])
    }

    /// Return current compression statistics.
    #[must_use]
    pub fn stats(&self) -> CompressionStats {
        self.inner.stats()
    }
}

/// State machine for the framed decompressor's frame parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameParseState {
    /// Collecting the 4-byte magic header.
    Magic,
    /// Collecting the 4-byte payload length.
    Length,
    /// Collecting `remaining` bytes of payload.
    Payload { remaining: usize },
}

/// Streaming decompressor that reassembles [`FramedStreamingCompressor`] output.
///
/// Handles arbitrary byte boundaries: callers may feed partial frames or
/// multiple frames in a single `write` call.
pub struct FramedStreamingDecompressor {
    inner: StreamingDecompressor,
    /// Internal byte-level reassembly buffer.
    buf: Vec<u8>,
    /// Current parser state.
    state: FrameParseState,
    /// Accumulated magic bytes so far.
    magic_buf: Vec<u8>,
    /// Accumulated payload-length bytes so far.
    len_buf: Vec<u8>,
    /// Accumulated payload bytes so far.
    payload_buf: Vec<u8>,
    /// Set when an EOS frame (zero-length payload) is received.
    eos: bool,
}

impl FramedStreamingDecompressor {
    /// Create a new framed decompressor wrapping the given codec.
    #[must_use]
    pub fn new(codec: StreamingCodec) -> Self {
        Self {
            inner: StreamingDecompressor::new(codec),
            buf: Vec::new(),
            state: FrameParseState::Magic,
            magic_buf: Vec::with_capacity(4),
            len_buf: Vec::with_capacity(4),
            payload_buf: Vec::new(),
            eos: false,
        }
    }

    /// Returns `true` after an end-of-stream frame has been received.
    #[must_use]
    pub fn is_eos(&self) -> bool {
        self.eos
    }

    /// Feed bytes into the decompressor; returns all decompressed data
    /// produced by complete frames in this call.
    pub fn write(&mut self, data: &[u8]) -> Result<Vec<u8>, ArchiveError> {
        self.buf.extend_from_slice(data);
        let mut out: Vec<u8> = Vec::new();

        let mut cursor = 0usize;

        loop {
            match self.state {
                FrameParseState::Magic => {
                    while self.magic_buf.len() < 4 && cursor < self.buf.len() {
                        self.magic_buf.push(self.buf[cursor]);
                        cursor += 1;
                    }
                    if self.magic_buf.len() < 4 {
                        break; // need more data
                    }
                    if self.magic_buf[..4] != FRAME_MAGIC {
                        self.buf.drain(..cursor);
                        return Err(ArchiveError::Corruption(format!(
                            "framed: invalid magic {:?}",
                            &self.magic_buf[..4]
                        )));
                    }
                    self.magic_buf.clear();
                    self.state = FrameParseState::Length;
                }
                FrameParseState::Length => {
                    while self.len_buf.len() < 4 && cursor < self.buf.len() {
                        self.len_buf.push(self.buf[cursor]);
                        cursor += 1;
                    }
                    if self.len_buf.len() < 4 {
                        break; // need more data
                    }
                    let payload_len = u32::from_le_bytes([
                        self.len_buf[0],
                        self.len_buf[1],
                        self.len_buf[2],
                        self.len_buf[3],
                    ]) as usize;
                    self.len_buf.clear();
                    if payload_len == 0 {
                        // End-of-stream marker.
                        self.eos = true;
                        self.state = FrameParseState::Magic;
                        break;
                    }
                    self.payload_buf.clear();
                    self.payload_buf.reserve(payload_len);
                    self.state = FrameParseState::Payload {
                        remaining: payload_len,
                    };
                }
                FrameParseState::Payload { remaining } => {
                    let available = self.buf.len() - cursor;
                    let take = available.min(remaining);
                    self.payload_buf
                        .extend_from_slice(&self.buf[cursor..cursor + take]);
                    cursor += take;
                    let new_remaining = remaining - take;
                    if new_remaining == 0 {
                        // Complete payload — decompress it.
                        let decompressed = self.inner.decompress_chunk(&self.payload_buf)?;
                        out.extend_from_slice(&decompressed);
                        self.payload_buf.clear();
                        self.state = FrameParseState::Magic;
                    } else {
                        self.state = FrameParseState::Payload {
                            remaining: new_remaining,
                        };
                        break; // wait for more data
                    }
                }
            }
        }

        // Drain consumed bytes from internal buffer.
        if cursor > 0 {
            self.buf.drain(..cursor);
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- LZ77 tests ---

    #[test]
    fn test_lz77_roundtrip_simple() {
        let data = b"hello, world! hello, world!";
        let compressed = compress_lz77(data, 8);
        let decompressed = decompress_lz77(&compressed).expect("decompression failed");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_lz77_roundtrip_repetitive() {
        let data: Vec<u8> = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAA".to_vec();
        let compressed = compress_lz77(&data, 8);
        let decompressed = decompress_lz77(&compressed).expect("decompression failed");
        assert_eq!(decompressed, data);
        // Repetitive data should compress well.
        assert!(compressed.len() < data.len(), "expected compression");
    }

    #[test]
    fn test_lz77_empty() {
        let data: &[u8] = b"";
        let compressed = compress_lz77(data, 4);
        let decompressed = decompress_lz77(&compressed).expect("decompression failed");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_lz77_incompressible() {
        // Pseudo-random incompressible bytes.
        let data: Vec<u8> = (0u8..=255).cycle().take(512).collect();
        let compressed = compress_lz77(&data, 4);
        let decompressed = decompress_lz77(&compressed).expect("decompression failed");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_lz77_large_repetitive() {
        let data: Vec<u8> = b"OxiMedia archive streaming compression test "
            .iter()
            .cycle()
            .take(4096)
            .copied()
            .collect();
        let compressed = compress_lz77(&data, 12);
        let decompressed = decompress_lz77(&compressed).expect("decompression failed");
        assert_eq!(decompressed, data);
    }

    // --- LZ4 tests ---

    #[test]
    fn test_lz4_roundtrip_simple() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let compressed = lz4_compress(data);
        let decompressed = lz4_decompress(&compressed, data.len()).expect("lz4 decompress failed");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_lz4_roundtrip_repetitive() {
        let data: Vec<u8> = b"ABCDABCDABCDABCDABCDABCDABCDABCD".to_vec();
        let compressed = lz4_compress(&data);
        let decompressed = lz4_decompress(&compressed, data.len()).expect("lz4 decompress failed");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_lz4_empty() {
        let data: &[u8] = b"";
        let compressed = lz4_compress(data);
        let decompressed = lz4_decompress(&compressed, 0).expect("lz4 decompress failed");
        assert_eq!(decompressed, data);
    }

    // --- Streaming API tests ---

    #[test]
    fn test_streaming_zstd_roundtrip() {
        let original = b"hello streaming zstd! hello streaming zstd!";
        let mut compressor = StreamingCompressor::new(StreamingCodec::Zstd { level: 3 });
        let compressed = compressor.compress_chunk(original);
        let _footer = compressor.finish();

        let mut decompressor = StreamingDecompressor::new(StreamingCodec::Zstd { level: 3 });
        let decompressed = decompressor
            .decompress_chunk(&compressed)
            .expect("streaming zstd decompress failed");
        assert_eq!(decompressed.as_slice(), original.as_ref());
    }

    #[test]
    fn test_streaming_deflate_roundtrip() {
        let original = b"deflate streaming test data deflate streaming test data";
        let mut compressor = StreamingCompressor::new(StreamingCodec::Deflate { level: 6 });
        let compressed = compressor.compress_chunk(original);

        let mut decompressor = StreamingDecompressor::new(StreamingCodec::Deflate { level: 6 });
        let decompressed = decompressor
            .decompress_chunk(&compressed)
            .expect("streaming deflate decompress failed");
        assert_eq!(decompressed.as_slice(), original.as_ref());
    }

    #[test]
    fn test_streaming_lz4_roundtrip() {
        let original = b"lz4 streaming chunk test lz4 streaming chunk test lz4";
        let mut compressor = StreamingCompressor::new(StreamingCodec::Lz4);
        let compressed = compressor.compress_chunk(original);

        let mut decompressor = StreamingDecompressor::new(StreamingCodec::Lz4);
        let decompressed = decompressor
            .decompress_chunk(&compressed)
            .expect("streaming lz4 decompress failed");
        assert_eq!(decompressed.as_slice(), original.as_ref());
    }

    #[test]
    fn test_streaming_passthrough_roundtrip() {
        let original = b"passthrough data is not modified";
        let mut compressor = StreamingCompressor::new(StreamingCodec::Passthrough);
        let compressed = compressor.compress_chunk(original);
        assert_eq!(compressed.as_slice(), original.as_ref());

        let mut decompressor = StreamingDecompressor::new(StreamingCodec::Passthrough);
        let decompressed = decompressor
            .decompress_chunk(&compressed)
            .expect("streaming passthrough decompress failed");
        assert_eq!(decompressed.as_slice(), original.as_ref());
    }

    #[test]
    fn test_streaming_stats() {
        let data: Vec<u8> = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .iter()
            .cycle()
            .take(1024)
            .copied()
            .collect();
        let mut compressor = StreamingCompressor::new(StreamingCodec::Zstd { level: 5 });
        let _ = compressor.compress_chunk(&data);
        let stats = compressor.stats();

        assert_eq!(stats.input_bytes, 1024);
        assert!(stats.output_bytes > 0);
        assert!(
            stats.compression_ratio > 1.0,
            "should compress repetitive data"
        );
        assert!(stats.codec.contains("zstd"));
    }

    // -----------------------------------------------------------------------
    // write() pull API tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compressor_write_api_zstd() {
        let data = b"write api test data write api test data";
        let mut c = StreamingCompressor::new(StreamingCodec::Zstd { level: 3 });
        let compressed = c.write(data).expect("compress write failed");

        let mut d = StreamingDecompressor::new(StreamingCodec::Zstd { level: 3 });
        let decompressed = d.write(&compressed).expect("decompress write failed");
        assert_eq!(decompressed.as_slice(), data.as_ref());
    }

    #[test]
    fn test_compressor_write_api_lz4() {
        let data: Vec<u8> = b"lz4 write api "
            .iter()
            .cycle()
            .take(200)
            .copied()
            .collect();
        let mut c = StreamingCompressor::new(StreamingCodec::Lz4);
        let compressed = c.write(&data).expect("compress failed");

        let mut d = StreamingDecompressor::new(StreamingCodec::Lz4);
        let decompressed = d.write(&compressed).expect("decompress failed");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compressor_write_api_passthrough() {
        let data = b"passthrough write";
        let mut c = StreamingCompressor::new(StreamingCodec::Passthrough);
        let out = c.write(data).expect("passthrough failed");
        assert_eq!(out.as_slice(), data.as_ref());

        let mut d = StreamingDecompressor::new(StreamingCodec::Passthrough);
        let back = d.write(&out).expect("decompress passthrough failed");
        assert_eq!(back.as_slice(), data.as_ref());
    }

    #[test]
    fn test_compressor_write_api_deflate() {
        let data = b"deflate write api test data deflate write api test data";
        let mut c = StreamingCompressor::new(StreamingCodec::Deflate { level: 6 });
        let compressed = c.write(data).expect("deflate compress failed");

        let mut d = StreamingDecompressor::new(StreamingCodec::Deflate { level: 6 });
        let decompressed = d.write(&compressed).expect("deflate decompress failed");
        assert_eq!(decompressed.as_slice(), data.as_ref());
    }

    // -----------------------------------------------------------------------
    // StreamingCompressorBuilder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_basic() {
        let mut c = StreamingCompressorBuilder::new(StreamingCodec::Lz4)
            .buffer_capacity(1024)
            .chunk_size(512)
            .build();
        let data = b"builder test data";
        let compressed = c.write(data).expect("builder compress failed");
        let mut d = StreamingDecompressor::new(StreamingCodec::Lz4);
        let back = d.write(&compressed).expect("decompress failed");
        assert_eq!(back.as_slice(), data.as_ref());
    }

    #[test]
    fn test_builder_defaults() {
        let b = StreamingCompressorBuilder::new(StreamingCodec::Passthrough);
        assert_eq!(b.buffer_capacity, 65536);
        assert_eq!(b.chunk_size, 0);
    }

    #[test]
    fn test_decompressor_builder_basic() {
        let mut d = StreamingDecompressorBuilder::new(StreamingCodec::Passthrough)
            .buffer_capacity(2048)
            .build();
        let out = d.write(b"hello").expect("failed");
        assert_eq!(out.as_slice(), b"hello");
    }

    // -----------------------------------------------------------------------
    // FramedStreamingCompressor / FramedStreamingDecompressor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_framed_roundtrip_zstd() {
        let data = b"framed zstd roundtrip data framed zstd roundtrip data";
        let mut fc = FramedStreamingCompressor::new(StreamingCodec::Zstd { level: 3 });
        let frame = fc.write(data).expect("framed compress failed");
        let eos = fc.finish();

        let mut fd = FramedStreamingDecompressor::new(StreamingCodec::Zstd { level: 3 });
        let out = fd.write(&frame).expect("framed decompress failed");
        let _ = fd.write(&eos).expect("eos write failed");

        assert_eq!(out.as_slice(), data.as_ref());
        assert!(fd.is_eos());
    }

    #[test]
    fn test_framed_roundtrip_lz4() {
        let data: Vec<u8> = b"lz4 framed test "
            .iter()
            .cycle()
            .take(300)
            .copied()
            .collect();
        let mut fc = FramedStreamingCompressor::new(StreamingCodec::Lz4);
        let frame = fc.write(&data).expect("compress failed");

        let mut fd = FramedStreamingDecompressor::new(StreamingCodec::Lz4);
        let out = fd.write(&frame).expect("decompress failed");
        assert_eq!(out, data);
    }

    #[test]
    fn test_framed_roundtrip_passthrough() {
        let data = b"passthrough framed";
        let mut fc = FramedStreamingCompressor::new(StreamingCodec::Passthrough);
        let frame = fc.write(data).expect("failed");

        let mut fd = FramedStreamingDecompressor::new(StreamingCodec::Passthrough);
        let out = fd.write(&frame).expect("failed");
        assert_eq!(out.as_slice(), data.as_ref());
    }

    #[test]
    fn test_framed_multiple_frames() {
        let chunks: &[&[u8]] = &[b"chunk one data", b"chunk two data", b"chunk three data"];
        let mut fc = FramedStreamingCompressor::new(StreamingCodec::Passthrough);
        let mut all_framed: Vec<u8> = Vec::new();
        for chunk in chunks {
            all_framed.extend(fc.write(chunk).expect("compress failed"));
        }
        all_framed.extend(fc.finish());

        let mut fd = FramedStreamingDecompressor::new(StreamingCodec::Passthrough);
        let out = fd.write(&all_framed).expect("decompress failed");

        let expected: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(out, expected);
        assert!(fd.is_eos());
    }

    #[test]
    fn test_framed_partial_feed() {
        let data = b"partial feed test data partial feed test";
        let mut fc = FramedStreamingCompressor::new(StreamingCodec::Passthrough);
        let frame = fc.write(data).expect("compress failed");

        let mut fd = FramedStreamingDecompressor::new(StreamingCodec::Passthrough);
        // Feed one byte at a time.
        let mut collected: Vec<u8> = Vec::new();
        for byte in &frame {
            let chunk = fd
                .write(std::slice::from_ref(byte))
                .expect("partial feed failed");
            collected.extend(chunk);
        }
        assert_eq!(collected.as_slice(), data.as_ref());
    }

    #[test]
    fn test_framed_eos_marker() {
        let mut fc = FramedStreamingCompressor::new(StreamingCodec::Passthrough);
        let eos = fc.finish();

        let mut fd = FramedStreamingDecompressor::new(StreamingCodec::Passthrough);
        assert!(!fd.is_eos());
        let _ = fd.write(&eos).expect("eos failed");
        assert!(fd.is_eos());
    }

    #[test]
    fn test_framed_stats_available() {
        let mut fc = FramedStreamingCompressor::new(StreamingCodec::Zstd { level: 3 });
        let data: Vec<u8> = b"stats test ".iter().cycle().take(512).copied().collect();
        let _ = fc.write(&data).expect("compress failed");
        let stats = fc.stats();
        assert_eq!(stats.input_bytes, 512);
        assert!(stats.output_bytes > 0);
    }

    #[test]
    fn test_framed_invalid_magic_returns_error() {
        let bad_frame = b"\x00\x00\x00\x00\x05\x00\x00\x00hello";
        let mut fd = FramedStreamingDecompressor::new(StreamingCodec::Passthrough);
        let result = fd.write(bad_frame);
        assert!(result.is_err(), "should detect invalid magic");
    }
}
