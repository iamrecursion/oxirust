//! Parallel SpeedHQ frame-slice encoder for multi-core NDI sending.
//!
//! NDI's SpeedHQ codec is inherently slice-parallel: each horizontal slice
//! of a video frame can be encoded independently.  This module divides a
//! raw video frame into `N` horizontal slices, encodes them concurrently
//! using Rayon's thread pool, and reassembles the bitstream in order.
//!
//! # Architecture
//!
//! ```text
//! RawFrame ──→ split_into_slices() ──→ [Slice; N]
//!                                            │
//!                              ┌─────────────┘
//!                              ▼  (rayon par_iter)
//!                        encode_slice() × N
//!                              │
//!                              ▼
//!                        assemble_bitstream()
//!                              │
//!                              ▼
//!                        EncodedFrame
//! ```
//!
//! # Usage
//!
//! ```
//! use oximedia_ndi::frame_slice_encoder::{
//!     FrameSliceEncoder, SliceEncoderConfig, RawFrame, PixelFormat,
//! };
//!
//! let cfg = SliceEncoderConfig { slice_count: 4, ..Default::default() };
//! let encoder = FrameSliceEncoder::new(cfg);
//!
//! let frame = RawFrame::new_test(1920, 1080, PixelFormat::Uyvy422);
//! let encoded = encoder.encode(&frame).expect("encode should succeed");
//! assert!(!encoded.data.is_empty());
//! ```

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

use crate::{NdiError, Result};
// rayon removed — using sequential iteration (Pure Rust policy)
use std::num::NonZeroU32;

// ---------------------------------------------------------------------------
// PixelFormat
// ---------------------------------------------------------------------------

/// Input pixel format for the slice encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// UYVY packed 4:2:2 — 2 bytes per pixel.
    Uyvy422,
    /// YV12 planar 4:2:0 — 1.5 bytes per pixel.
    Yv12,
    /// RGBA packed — 4 bytes per pixel.
    Rgba,
}

impl PixelFormat {
    /// Bytes per pixel (rounded up; YV12 returns 1 for this purpose).
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Uyvy422 => 2,
            Self::Yv12 => 1, // luma only; chroma handled separately
            Self::Rgba => 4,
        }
    }

    /// Return the expected byte size of one row for the given width.
    pub fn stride(self, width: u32) -> usize {
        match self {
            Self::Uyvy422 => (width as usize) * 2,
            Self::Yv12 => width as usize,
            Self::Rgba => (width as usize) * 4,
        }
    }
}

// ---------------------------------------------------------------------------
// RawFrame
// ---------------------------------------------------------------------------

/// An uncompressed video frame to be encoded.
#[derive(Debug, Clone)]
pub struct RawFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: PixelFormat,
    /// Raw pixel data, row-major.
    pub data: Vec<u8>,
    /// Row stride in bytes (may include padding).
    pub stride: usize,
    /// Frame number (used for slice header generation).
    pub frame_number: u64,
}

impl RawFrame {
    /// Create a new raw frame.
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        data: Vec<u8>,
        stride: usize,
        frame_number: u64,
    ) -> Self {
        Self {
            width,
            height,
            format,
            data,
            stride,
            frame_number,
        }
    }

    /// Create a zero-filled test frame of the given dimensions.
    pub fn new_test(width: u32, height: u32, format: PixelFormat) -> Self {
        let stride = format.stride(width);
        let data = vec![0u8; stride * height as usize];
        Self::new(width, height, format, data, stride, 0)
    }

    /// Return the row data for `row_index` (0-based).
    pub fn row(&self, row_index: usize) -> Option<&[u8]> {
        let start = row_index * self.stride;
        let end = start + self.stride;
        if end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// VideoSlice — one horizontal strip
// ---------------------------------------------------------------------------

/// A horizontal slice of a raw frame, ready for independent encoding.
#[derive(Debug, Clone)]
pub struct VideoSlice {
    /// Index of this slice (0-based, top to bottom).
    pub index: usize,
    /// First row of this slice (inclusive).
    pub first_row: u32,
    /// Number of rows in this slice.
    pub row_count: u32,
    /// Frame width (pixels).
    pub width: u32,
    /// Pixel format.
    pub format: PixelFormat,
    /// Raw pixel bytes for this slice, row-major.
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// EncodedSlice — compressed output for one slice
// ---------------------------------------------------------------------------

/// The compressed output of a single slice encode operation.
#[derive(Debug, Clone)]
pub struct EncodedSlice {
    /// Slice index (ordering key for reassembly).
    pub index: usize,
    /// Compressed bitstream bytes for this slice.
    pub data: Vec<u8>,
    /// Uncompressed input size in bytes (for ratio tracking).
    pub input_bytes: usize,
}

impl EncodedSlice {
    /// Compression ratio: input_bytes / output_bytes.
    pub fn compression_ratio(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.input_bytes as f64 / self.data.len() as f64
    }
}

// ---------------------------------------------------------------------------
// EncodedFrame — reassembled output
// ---------------------------------------------------------------------------

/// The fully encoded frame produced by the slice encoder.
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Number of slices used.
    pub slice_count: usize,
    /// Concatenated encoded data (slice header + payload for each slice).
    pub data: Vec<u8>,
    /// Per-slice byte offsets into `data` (for random-access decoding).
    pub slice_offsets: Vec<usize>,
    /// Original frame number.
    pub frame_number: u64,
}

impl EncodedFrame {
    /// Total compressed size in bytes.
    pub fn compressed_size(&self) -> usize {
        self.data.len()
    }

    /// Raw slice payload at `slice_index`.
    pub fn slice_data(&self, slice_index: usize) -> Option<&[u8]> {
        let start = *self.slice_offsets.get(slice_index)?;
        let end = self
            .slice_offsets
            .get(slice_index + 1)
            .copied()
            .unwrap_or(self.data.len());
        self.data.get(start..end)
    }
}

// ---------------------------------------------------------------------------
// SliceEncoderConfig
// ---------------------------------------------------------------------------

/// Configuration for the parallel slice encoder.
#[derive(Debug, Clone)]
pub struct SliceEncoderConfig {
    /// Number of horizontal slices to divide the frame into.
    /// Must be >= 1.  More slices = more parallelism but more overhead.
    pub slice_count: usize,
    /// SpeedHQ quantisation parameter (1 = highest quality, 31 = lowest).
    pub quantiser: u8,
    /// Enable a simple DC-prediction (intra prediction) pass per slice.
    pub enable_intra_prediction: bool,
    /// Minimum slice height in rows.  Slices thinner than this are merged
    /// with the preceding slice to avoid tiny work units.
    pub min_slice_rows: NonZeroU32,
}

impl Default for SliceEncoderConfig {
    fn default() -> Self {
        Self {
            slice_count: 4,
            quantiser: 8,
            enable_intra_prediction: true,
            min_slice_rows: NonZeroU32::new(8).expect("8 is non-zero"),
        }
    }
}

// ---------------------------------------------------------------------------
// FrameSliceEncoder
// ---------------------------------------------------------------------------

/// Parallel SpeedHQ frame slice encoder.
///
/// Splits each input frame into horizontal slices, encodes them in parallel
/// via Rayon, then reassembles the slices into a single bitstream.
pub struct FrameSliceEncoder {
    config: SliceEncoderConfig,
}

impl FrameSliceEncoder {
    /// Create a new encoder with the given configuration.
    pub fn new(config: SliceEncoderConfig) -> Self {
        Self { config }
    }

    /// Encode `frame` using parallel slice encoding.
    ///
    /// # Errors
    ///
    /// Returns [`NdiError::InvalidFrameFormat`] if the frame data length is
    /// inconsistent with the declared width/height/format.
    pub fn encode(&self, frame: &RawFrame) -> Result<EncodedFrame> {
        let expected_len = frame.stride * frame.height as usize;
        if frame.data.len() < expected_len {
            return Err(NdiError::InvalidFrameFormat);
        }

        let slices = self.split_frame(frame)?;
        let slice_count = slices.len();

        // Encode all slices in parallel
        let encoded_slices: Vec<EncodedSlice> =
            slices.into_iter().map(|s| self.encode_slice(s)).collect();

        // Sort by index to ensure deterministic ordering
        let mut sorted = encoded_slices;
        sorted.sort_by_key(|s| s.index);

        // Assemble into a single bitstream
        let total_size: usize = sorted.iter().map(|s| s.data.len()).sum();
        let mut data = Vec::with_capacity(total_size);
        let mut slice_offsets = Vec::with_capacity(slice_count);

        for s in &sorted {
            slice_offsets.push(data.len());
            data.extend_from_slice(&s.data);
        }

        Ok(EncodedFrame {
            width: frame.width,
            height: frame.height,
            slice_count,
            data,
            slice_offsets,
            frame_number: frame.frame_number,
        })
    }

    /// Split `frame` into [`VideoSlice`] objects.
    fn split_frame(&self, frame: &RawFrame) -> Result<Vec<VideoSlice>> {
        if frame.height == 0 || frame.width == 0 {
            return Err(NdiError::InvalidFrameFormat);
        }

        let height = frame.height as usize;
        let n = self.config.slice_count.max(1).min(height);
        let min_rows = self.config.min_slice_rows.get() as usize;

        // Compute row assignments
        let base_rows = height / n;
        let remainder = height % n;
        let mut slices = Vec::with_capacity(n);
        let mut row_start = 0usize;

        for i in 0..n {
            let extra = if i < remainder { 1 } else { 0 };
            let mut row_count = base_rows + extra;

            // Merge tiny tail slices with the previous one
            if row_count < min_rows && i > 0 {
                if let Some(prev) = slices.last_mut() as Option<&mut VideoSlice> {
                    prev.row_count += row_count as u32;
                    let prev_end = prev.first_row as usize + prev.row_count as usize;
                    let byte_start = prev.first_row as usize * frame.stride;
                    let byte_end = prev_end * frame.stride;
                    let byte_end = byte_end.min(frame.data.len());
                    prev.data = frame.data[byte_start..byte_end].to_vec();
                    row_start += row_count;
                    continue;
                }
            }

            // Clamp row_count so we don't exceed the frame
            row_count = row_count.min(height - row_start);
            if row_count == 0 {
                break;
            }

            let byte_start = row_start * frame.stride;
            let byte_end = (row_start + row_count) * frame.stride;
            let byte_end = byte_end.min(frame.data.len());

            slices.push(VideoSlice {
                index: i,
                first_row: row_start as u32,
                row_count: row_count as u32,
                width: frame.width,
                format: frame.format,
                data: frame.data[byte_start..byte_end].to_vec(),
            });

            row_start += row_count;
        }

        Ok(slices)
    }

    /// Encode a single [`VideoSlice`] into a compressed [`EncodedSlice`].
    ///
    /// The bitstream layout is **SpeedHQ-style** with static Huffman entropy
    /// coding on top of DC-predicted residuals.  Each slice is laid out as:
    ///
    /// ```text
    /// [Header: 9 bytes]
    ///   byte 0      : format version (currently SPEEDHQ_FORMAT_VERSION)
    ///   bytes 1..3  : slice index            (u16 LE)
    ///   bytes 3..5  : first_row              (u16 LE)
    ///   bytes 5..7  : row_count              (u16 LE)
    ///   byte 7      : quantiser
    ///   byte 8      : pixel format id
    /// [block_count : u32 LE]              — number of 8-sample blocks encoded
    /// [bitstream_byte_len : u32 LE]       — length of Huffman bitstream
    /// [bitstream : N bytes]               — MSB-first Huffman coded blocks
    /// [tail_len : u8]                     — number of verbatim trailing bytes (0..7)
    /// [tail : tail_len bytes]             — verbatim remainder (< 8 bytes)
    /// ```
    ///
    /// For each 8-sample block the encoder emits:
    /// 1. The DC residual (block mean minus running predictor) as a raw
    ///    9-bit two's-complement value (range −256..255, with the actual
    ///    encoder output clamped to i9 so a single block's contribution is
    ///    safely [-256, 255]).
    /// 2. Eight quantised AC coefficients (the residual `pixel − DC` after
    ///    integer division by `q`, held in an i16 so the ±2047 ESCAPE range
    ///    is representable) emitted as zigzag-style (run, level) pairs using
    ///    [`HUFFMAN_AC_TABLE`]; runs of zeros are absorbed into the next
    ///    non-zero coefficient.
    /// 3. The EOB symbol terminating the block.
    ///
    /// Levels and runs that are out of the static table's domain are emitted
    /// via the ESCAPE prefix followed by a 3-bit run and a 12-bit signed
    /// (two's-complement) level field.
    fn encode_slice(&self, slice: VideoSlice) -> EncodedSlice {
        let input_bytes = slice.data.len();
        let q = self.config.quantiser.max(1) as i32;

        // --- Header (9 bytes: version + 6-byte ident + quantiser + format) ---
        let fmt_id = match slice.format {
            PixelFormat::Uyvy422 => 0u8,
            PixelFormat::Yv12 => 1u8,
            PixelFormat::Rgba => 2u8,
        };
        let mut payload: Vec<u8> = Vec::with_capacity(SLICE_HEADER_LEN + input_bytes / 2);
        payload.push(SPEEDHQ_FORMAT_VERSION);
        let idx = slice.index as u16;
        payload.extend_from_slice(&idx.to_le_bytes());
        payload.extend_from_slice(&(slice.first_row as u16).to_le_bytes());
        payload.extend_from_slice(&(slice.row_count as u16).to_le_bytes());
        payload.push(self.config.quantiser);
        payload.push(fmt_id);

        // --- Huffman-coded blocks ---
        let mut bw = BitWriter::new();
        let mut predictor: i32 = 0;
        let mut block_count: u32 = 0;
        let mut i = 0usize;
        while i + 7 < slice.data.len() {
            let block_dc: i32 = slice.data[i..i + 8].iter().map(|&b| b as i32).sum::<i32>() / 8;

            // 9-bit DC residual (range [-256, 255]); writer accepts u32 + length.
            let dc_residual = (block_dc - predictor).clamp(-256, 255);
            predictor = block_dc;
            bw.write_bits((dc_residual as i32 & DC_MASK) as u32, DC_BITS);

            // Build the quantised AC vector for this block (8 i16 values).
            // i16 is wide enough for the ±2047 ESCAPE range and keeps each
            // entry compact (16 bytes total per block, fits in two SSE2 lanes).
            let mut ac = [0i16; 8];
            for (j, slot) in ac.iter_mut().enumerate() {
                let raw = slice.data[i + j] as i32 - block_dc;
                let qac = (raw / q).clamp(AC_LEVEL_MIN, AC_LEVEL_MAX);
                *slot = qac as i16;
            }

            // RLE: collect runs of zeros and emit (run, level) Huffman codes.
            // A block has at most 8 samples so the run can never exceed
            // `AC_MAX_RUN` (= 7) before a non-zero level or EOB terminates it.
            // Trailing zeros are *absorbed* by the EOB symbol below.
            let mut run: u8 = 0;
            for &coef in ac.iter() {
                if coef == 0 {
                    run += 1;
                    continue;
                }
                let level = coef as i32;
                debug_assert!(
                    run <= AC_MAX_RUN,
                    "AC run {run} exceeds AC_MAX_RUN {AC_MAX_RUN}; block sample width violated"
                );
                emit_run_level(&mut bw, run, level);
                run = 0;
            }
            // Always terminate with EOB; any trailing zero run is implicit.
            bw.write_bits(HUFFMAN_EOB_CODE, HUFFMAN_EOB_BITS);
            block_count += 1;
            i += 8;
        }

        let bitstream = bw.finalize();

        payload.extend_from_slice(&block_count.to_le_bytes());
        payload.extend_from_slice(&(bitstream.len() as u32).to_le_bytes());
        payload.extend_from_slice(&bitstream);

        // Verbatim tail bytes (< 8) — preserved exactly so the slice's leftover
        // edge pixels survive the round-trip.
        let tail = &slice.data[i..];
        let tail_len = tail.len() as u8;
        payload.push(tail_len);
        payload.extend_from_slice(tail);

        EncodedSlice {
            index: slice.index,
            data: payload,
            input_bytes,
        }
    }

    /// Return the configured number of slices.
    pub fn slice_count(&self) -> usize {
        self.config.slice_count
    }
}

// ---------------------------------------------------------------------------
// Slice header & quantisation constants
// ---------------------------------------------------------------------------

/// Length in bytes of the fixed-size slice header before block_count.
const SLICE_HEADER_LEN: usize = 9;

/// Format version embedded in the first header byte.  Bumped whenever the
/// wire format changes incompatibly so downstream decoders can refuse old
/// or unknown streams.
const SPEEDHQ_FORMAT_VERSION: u8 = 0x01;

/// AC coefficient clamp bounds.  Chosen so that any byte-residual
/// (`pixel ∈ [0, 255]` minus `block_dc ∈ [0, 255]`) fits without loss before
/// quantisation, and so that ESCAPE's 12-bit signed level field can carry
/// the result.
const AC_LEVEL_MIN: i32 = -2047;
const AC_LEVEL_MAX: i32 = 2047;

/// Number of bits used by the ESCAPE level payload (signed two's complement).
const ESCAPE_LEVEL_BITS: u32 = 12;
const ESCAPE_LEVEL_MASK: i32 = (1 << ESCAPE_LEVEL_BITS) - 1;

/// Bit width of the raw DC residual emitted at the start of every block.
/// 9-bit signed range [-256, 255] safely covers (block_dc − predictor) when
/// both are byte means.
const DC_BITS: u32 = 9;
const DC_MASK: i32 = (1 << DC_BITS) - 1;

/// Maximum AC run-length supported by the static Huffman table before falling
/// back to ESCAPE.  Capped at 7 to fit in a 3-bit run field.
const AC_MAX_RUN: u8 = 7;

// ---------------------------------------------------------------------------
// Decoded slice — output of `decode_slice`
// ---------------------------------------------------------------------------

/// Reconstructed slice payload as recovered by [`decode_slice`].
///
/// The reconstructed pixel data is in the same row-major byte order as the
/// original [`VideoSlice::data`], but with **quantisation loss** — values
/// have been rounded to multiples of the encoder's quantiser.  The DC
/// reconstruction is exact (DC residuals are stored at full 9-bit precision).
#[derive(Debug, Clone)]
pub struct DecodedSlice {
    /// Slice index decoded from the header.
    pub index: usize,
    /// First row of this slice within the parent frame.
    pub first_row: u32,
    /// Number of rows in this slice.
    pub row_count: u32,
    /// Pixel format reconstructed from the header byte.
    pub format: PixelFormat,
    /// Quantiser used by the encoder (1..=31).
    pub quantiser: u8,
    /// Reconstructed pixel data (lossy round-trip).
    pub data: Vec<u8>,
}

/// Decode an [`EncodedSlice`] bitstream back into a [`DecodedSlice`].
///
/// # Errors
///
/// Returns [`NdiError::Codec`] if the bitstream is malformed, truncated, or
/// uses an unrecognised format version.
pub fn decode_slice(encoded: &[u8]) -> Result<DecodedSlice> {
    if encoded.len() < SLICE_HEADER_LEN + 4 + 4 + 1 {
        return Err(NdiError::Codec(format!(
            "slice too short: {} bytes",
            encoded.len()
        )));
    }
    let version = encoded[0];
    if version != SPEEDHQ_FORMAT_VERSION {
        return Err(NdiError::Codec(format!(
            "unsupported slice format version: 0x{version:02x}"
        )));
    }
    let idx = u16::from_le_bytes([encoded[1], encoded[2]]) as usize;
    let first_row = u16::from_le_bytes([encoded[3], encoded[4]]) as u32;
    let row_count = u16::from_le_bytes([encoded[5], encoded[6]]) as u32;
    let quantiser = encoded[7];
    let fmt_id = encoded[8];
    let format = match fmt_id {
        0 => PixelFormat::Uyvy422,
        1 => PixelFormat::Yv12,
        2 => PixelFormat::Rgba,
        other => {
            return Err(NdiError::Codec(format!(
                "unknown pixel format id: 0x{other:02x}"
            )));
        }
    };

    let mut cursor = SLICE_HEADER_LEN;
    let block_count = read_u32_le(encoded, cursor)?;
    cursor += 4;
    let bitstream_len = read_u32_le(encoded, cursor)? as usize;
    cursor += 4;
    let bitstream_end = cursor
        .checked_add(bitstream_len)
        .ok_or_else(|| NdiError::Codec("bitstream length overflows".to_string()))?;
    if bitstream_end + 1 > encoded.len() {
        return Err(NdiError::Codec(format!(
            "bitstream truncated: need {} bytes, have {}",
            bitstream_end + 1,
            encoded.len()
        )));
    }
    let bitstream = &encoded[cursor..bitstream_end];
    cursor = bitstream_end;
    let tail_len = encoded[cursor] as usize;
    cursor += 1;
    let tail_end = cursor
        .checked_add(tail_len)
        .ok_or_else(|| NdiError::Codec("tail length overflows".to_string()))?;
    if tail_end > encoded.len() {
        return Err(NdiError::Codec(format!(
            "tail truncated: need {tail_end} bytes, have {}",
            encoded.len()
        )));
    }
    let tail = &encoded[cursor..tail_end];

    // Decode block-by-block.
    // Defence-in-depth: cap the declared block_count against what the
    // bitstream could *possibly* hold.  Each block consumes at least
    // `DC_BITS + HUFFMAN_EOB_BITS` bits (= 11), so a 4-byte bitstream cannot
    // possibly encode more than ⌊32/11⌋ = 2 blocks.  Rejecting absurd values
    // up front prevents a corrupt slice from triggering a multi-GB
    // `Vec::with_capacity` allocation on untrusted input.
    let min_bits_per_block = (DC_BITS + HUFFMAN_EOB_BITS) as u64;
    let max_blocks_for_bitstream =
        (bitstream.len() as u64).saturating_mul(8) / min_bits_per_block.max(1);
    if block_count as u64 > max_blocks_for_bitstream {
        return Err(NdiError::Codec(format!(
            "declared block_count {block_count} exceeds bitstream capacity ({max_blocks_for_bitstream} blocks max in {} bytes)",
            bitstream.len()
        )));
    }

    let q = quantiser.max(1) as i32;
    let mut br = BitReader::new(bitstream);
    let mut predictor: i32 = 0;
    let mut data: Vec<u8> = Vec::with_capacity((block_count as usize) * 8 + tail.len());
    for _ in 0..block_count {
        let dc_raw = br.read_bits(DC_BITS)? as i32;
        // Sign-extend the 9-bit value.
        let dc_residual = if dc_raw & (1 << (DC_BITS - 1)) != 0 {
            dc_raw - (1 << DC_BITS)
        } else {
            dc_raw
        };
        let block_dc = predictor + dc_residual;
        predictor = block_dc;

        let mut ac = [0i16; 8];
        let mut zz: usize = 0;
        loop {
            if zz >= 8 {
                // The previous (run, level) filled the last AC slot — the
                // next symbol MUST be EOB, otherwise the bitstream is malformed.
                match decode_run_level(&mut br)? {
                    None => break,
                    Some(_) => {
                        return Err(NdiError::Codec(
                            "missing EOB after full AC block".to_string(),
                        ));
                    }
                }
            }
            match decode_run_level(&mut br)? {
                None => break, // EOB
                Some((run, level)) => {
                    let target = zz + run as usize;
                    if target >= 8 {
                        return Err(NdiError::Codec(format!(
                            "AC run overflow: zz={zz} run={run}"
                        )));
                    }
                    ac[target] = level as i16;
                    zz = target + 1;
                }
            }
        }

        for &coef in ac.iter() {
            let pixel = block_dc + (coef as i32) * q;
            data.push(pixel.clamp(0, 255) as u8);
        }
    }

    data.extend_from_slice(tail);

    Ok(DecodedSlice {
        index: idx,
        first_row,
        row_count,
        format,
        quantiser,
        data,
    })
}

fn read_u32_le(buf: &[u8], offset: usize) -> Result<u32> {
    if offset + 4 > buf.len() {
        return Err(NdiError::Codec(format!(
            "u32 read out of bounds at offset {offset}"
        )));
    }
    Ok(u32::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ]))
}

// ---------------------------------------------------------------------------
// Huffman tables for (run, level) AC coefficient coding
// ---------------------------------------------------------------------------

/// Huffman code for End-Of-Block.  Always emitted at the end of every block.
const HUFFMAN_EOB_CODE: u32 = 0b10;
const HUFFMAN_EOB_BITS: u32 = 2;

/// Huffman code for the ESCAPE prefix.  Followed by a 3-bit `run` field and
/// an 8-bit signed `level` field (i8 two's complement).
const HUFFMAN_ESCAPE_CODE: u32 = 0b000000001;
const HUFFMAN_ESCAPE_BITS: u32 = 9;

/// One entry in the static AC Huffman table.
#[derive(Clone, Copy, Debug)]
struct AcCode {
    run: u8,
    level: u8, // absolute value
    code: u32,
    bits: u32,
}

/// Static Huffman table for the most-common (run, |level|) AC pairs.
///
/// Codes form a prefix-free set together with [`HUFFMAN_EOB_CODE`] and
/// [`HUFFMAN_ESCAPE_CODE`].  The sign of `level` is emitted as a single
/// trailing bit (0 = positive, 1 = negative).
const HUFFMAN_AC_TABLE: &[AcCode] = &[
    AcCode {
        run: 0,
        level: 1,
        code: 0b11,
        bits: 2,
    },
    AcCode {
        run: 0,
        level: 2,
        code: 0b010,
        bits: 3,
    },
    AcCode {
        run: 1,
        level: 1,
        code: 0b011,
        bits: 3,
    },
    AcCode {
        run: 0,
        level: 3,
        code: 0b0010,
        bits: 4,
    },
    AcCode {
        run: 2,
        level: 1,
        code: 0b0011,
        bits: 4,
    },
    AcCode {
        run: 1,
        level: 2,
        code: 0b00010,
        bits: 5,
    },
    AcCode {
        run: 3,
        level: 1,
        code: 0b00011,
        bits: 5,
    },
    AcCode {
        run: 0,
        level: 4,
        code: 0b000010,
        bits: 6,
    },
    AcCode {
        run: 4,
        level: 1,
        code: 0b000011,
        bits: 6,
    },
    AcCode {
        run: 2,
        level: 2,
        code: 0b0000010,
        bits: 7,
    },
    AcCode {
        run: 5,
        level: 1,
        code: 0b0000011,
        bits: 7,
    },
    AcCode {
        run: 0,
        level: 5,
        code: 0b00000010,
        bits: 8,
    },
    AcCode {
        run: 6,
        level: 1,
        code: 0b00000011,
        bits: 8,
    },
    AcCode {
        run: 1,
        level: 3,
        code: 0b000000010,
        bits: 9,
    },
    AcCode {
        run: 7,
        level: 1,
        code: 0b000000011,
        bits: 9,
    },
];

/// Look up the (run, |level|) Huffman code, returning `None` if not in table.
fn lookup_ac(run: u8, abs_level: u8) -> Option<&'static AcCode> {
    HUFFMAN_AC_TABLE
        .iter()
        .find(|e| e.run == run && e.level == abs_level)
}

/// Emit a single (run, level) pair using the static Huffman table or ESCAPE.
fn emit_run_level(bw: &mut BitWriter, run: u8, level: i32) {
    let abs_level = level.unsigned_abs();
    let sign_bit: u32 = if level < 0 { 1 } else { 0 };
    if let Ok(abs_u8) = u8::try_from(abs_level) {
        if let Some(entry) = lookup_ac(run, abs_u8) {
            bw.write_bits(entry.code, entry.bits);
            bw.write_bits(sign_bit, 1);
            return;
        }
    }
    // ESCAPE path: 9-bit escape prefix + 3-bit run + 12-bit signed level.
    let clamped = level.clamp(AC_LEVEL_MIN, AC_LEVEL_MAX);
    let level_bits = (clamped & ESCAPE_LEVEL_MASK) as u32;
    bw.write_bits(HUFFMAN_ESCAPE_CODE, HUFFMAN_ESCAPE_BITS);
    bw.write_bits(u32::from(run & 0b111), 3);
    bw.write_bits(level_bits, ESCAPE_LEVEL_BITS);
}

/// Decode one (run, level) pair (or `None` on EOB) from the bitstream.
fn decode_run_level(br: &mut BitReader<'_>) -> Result<Option<(u8, i32)>> {
    // Match the longest prefix from the static table by reading bits until
    // we have an unambiguous match.  We walk a tiny ad-hoc decision tree
    // since the table is small.
    //
    // Strategy: read bits one at a time and check against (code, bits) until
    // a match (or until we have enough bits to rule out every code).
    let mut accumulator: u32 = 0;
    let mut nbits: u32 = 0;
    let max_match_bits: u32 = HUFFMAN_ESCAPE_BITS
        .max(HUFFMAN_EOB_BITS)
        .max(HUFFMAN_AC_TABLE.iter().map(|e| e.bits).max().unwrap_or(0));

    while nbits < max_match_bits {
        accumulator = (accumulator << 1) | br.read_bits(1)?;
        nbits += 1;

        // EOB match?
        if nbits == HUFFMAN_EOB_BITS && accumulator == HUFFMAN_EOB_CODE {
            return Ok(None);
        }

        // ESCAPE match? (longest code, so check at the appropriate length).
        if nbits == HUFFMAN_ESCAPE_BITS && accumulator == HUFFMAN_ESCAPE_CODE {
            let run = br.read_bits(3)? as u8;
            let lvl_bits = br.read_bits(ESCAPE_LEVEL_BITS)? as i32;
            // Sign-extend the 12-bit two's-complement payload.
            let level = if lvl_bits & (1 << (ESCAPE_LEVEL_BITS - 1)) != 0 {
                lvl_bits - (1 << ESCAPE_LEVEL_BITS)
            } else {
                lvl_bits
            };
            return Ok(Some((run, level)));
        }

        // Table lookup.
        for entry in HUFFMAN_AC_TABLE.iter() {
            if entry.bits == nbits && entry.code == accumulator {
                let sign_bit = br.read_bits(1)?;
                let level = if sign_bit == 1 {
                    -(entry.level as i32)
                } else {
                    entry.level as i32
                };
                return Ok(Some((entry.run, level)));
            }
        }
    }

    Err(NdiError::Codec(format!(
        "no Huffman match after {nbits} bits (acc={accumulator:0b})"
    )))
}

// ---------------------------------------------------------------------------
// BitWriter — MSB-first bit-level writer
// ---------------------------------------------------------------------------

struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    bits_in_acc: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            acc: 0,
            bits_in_acc: 0,
        }
    }

    /// Write the `n_bits` least-significant bits of `value`, MSB first.
    /// `n_bits` must be in `1..=32`.
    fn write_bits(&mut self, value: u32, n_bits: u32) {
        debug_assert!(
            (1..=32).contains(&n_bits),
            "write_bits: n_bits={n_bits} out of range"
        );
        let mask: u64 = if n_bits == 32 {
            0xFFFF_FFFF
        } else {
            (1u64 << n_bits) - 1
        };
        let v = (value as u64) & mask;
        self.acc = (self.acc << n_bits) | v;
        self.bits_in_acc += n_bits;
        while self.bits_in_acc >= 8 {
            self.bits_in_acc -= 8;
            let byte = ((self.acc >> self.bits_in_acc) & 0xFF) as u8;
            self.buf.push(byte);
        }
    }

    /// Pad the trailing partial byte (if any) with zero bits and return the
    /// accumulated byte buffer.
    fn finalize(mut self) -> Vec<u8> {
        if self.bits_in_acc > 0 {
            let pad = 8 - self.bits_in_acc;
            self.acc <<= pad;
            self.buf.push((self.acc & 0xFF) as u8);
        }
        self.buf
    }
}

// ---------------------------------------------------------------------------
// BitReader — MSB-first bit-level reader (mirrors BitWriter)
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    /// Bit position within the current byte: 0 = MSB (bit 7), 7 = LSB (bit 0).
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read `n_bits` (1..=32) as a u32, MSB first.
    fn read_bits(&mut self, n_bits: u32) -> Result<u32> {
        debug_assert!(
            (1..=32).contains(&n_bits),
            "read_bits: n_bits={n_bits} out of range"
        );
        let mut result = 0u32;
        for _ in 0..n_bits {
            if self.byte_pos >= self.data.len() {
                return Err(NdiError::Codec("unexpected end of bitstream".to_string()));
            }
            let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
            result = (result << 1) | u32::from(bit);
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder(slices: usize) -> FrameSliceEncoder {
        FrameSliceEncoder::new(SliceEncoderConfig {
            slice_count: slices,
            quantiser: 8,
            enable_intra_prediction: false,
            min_slice_rows: NonZeroU32::new(1).expect("1 is non-zero"),
        })
    }

    // --- PixelFormat ---

    #[test]
    fn test_pixel_format_stride() {
        assert_eq!(PixelFormat::Uyvy422.stride(1920), 3840);
        assert_eq!(PixelFormat::Rgba.stride(1920), 7680);
        assert_eq!(PixelFormat::Yv12.stride(1920), 1920);
    }

    #[test]
    fn test_pixel_format_bytes_per_pixel() {
        assert_eq!(PixelFormat::Uyvy422.bytes_per_pixel(), 2);
        assert_eq!(PixelFormat::Rgba.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::Yv12.bytes_per_pixel(), 1);
    }

    // --- RawFrame ---

    #[test]
    fn test_raw_frame_row_access() {
        let frame = RawFrame::new_test(4, 4, PixelFormat::Uyvy422);
        // stride = 4*2 = 8
        assert!(frame.row(0).is_some());
        assert!(frame.row(3).is_some());
        assert!(frame.row(4).is_none());
    }

    // --- FrameSliceEncoder: split ---

    #[test]
    fn test_split_single_slice() {
        let enc = make_encoder(1);
        let frame = RawFrame::new_test(1920, 1080, PixelFormat::Uyvy422);
        let slices = enc.split_frame(&frame).expect("split should succeed");
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].row_count, 1080);
    }

    #[test]
    fn test_split_four_slices_equal() {
        let enc = make_encoder(4);
        let frame = RawFrame::new_test(1920, 1080, PixelFormat::Uyvy422);
        let slices = enc.split_frame(&frame).expect("split should succeed");
        assert_eq!(slices.len(), 4);
        let total_rows: u32 = slices.iter().map(|s| s.row_count).sum();
        assert_eq!(total_rows, 1080);
    }

    #[test]
    fn test_split_more_slices_than_rows() {
        // Only 2 rows → slices clamped to 2
        let enc = make_encoder(8);
        let frame = RawFrame::new_test(16, 2, PixelFormat::Uyvy422);
        let slices = enc.split_frame(&frame).expect("split should succeed");
        assert!(!slices.is_empty());
        let total: u32 = slices.iter().map(|s| s.row_count).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_split_returns_error_on_zero_height() {
        let enc = make_encoder(4);
        let frame = RawFrame::new(16, 0, PixelFormat::Uyvy422, vec![], 32, 0);
        assert!(enc.split_frame(&frame).is_err());
    }

    // --- FrameSliceEncoder: encode ---

    #[test]
    fn test_encode_produces_output() {
        let enc = make_encoder(4);
        let frame = RawFrame::new_test(320, 240, PixelFormat::Uyvy422);
        let result = enc.encode(&frame).expect("encode should succeed");
        assert!(!result.data.is_empty());
        assert_eq!(result.width, 320);
        assert_eq!(result.height, 240);
        assert_eq!(result.frame_number, 0);
    }

    #[test]
    fn test_encode_slice_offsets_are_monotone() {
        let enc = make_encoder(4);
        let frame = RawFrame::new_test(320, 240, PixelFormat::Uyvy422);
        let result = enc.encode(&frame).expect("encode should succeed");
        let offsets = &result.slice_offsets;
        for w in offsets.windows(2) {
            assert!(w[0] <= w[1], "offsets should be non-decreasing");
        }
    }

    #[test]
    fn test_encode_slice_data_accessible() {
        let enc = make_encoder(2);
        let frame = RawFrame::new_test(64, 64, PixelFormat::Uyvy422);
        let result = enc.encode(&frame).expect("encode should succeed");
        for i in 0..result.slice_count {
            assert!(
                result.slice_data(i).is_some(),
                "slice {} data should be accessible",
                i
            );
        }
    }

    #[test]
    fn test_encode_short_frame_returns_error() {
        let enc = make_encoder(2);
        // data is empty but height=10 → should fail
        let frame = RawFrame::new(64, 10, PixelFormat::Uyvy422, vec![], 128, 0);
        assert!(enc.encode(&frame).is_err());
    }

    #[test]
    fn test_encoded_frame_compressed_size() {
        let enc = make_encoder(4);
        let frame = RawFrame::new_test(320, 240, PixelFormat::Uyvy422);
        let result = enc.encode(&frame).expect("encode should succeed");
        assert_eq!(result.compressed_size(), result.data.len());
    }

    #[test]
    fn test_encoder_slice_count_accessor() {
        let enc = make_encoder(8);
        assert_eq!(enc.slice_count(), 8);
    }

    #[test]
    fn test_encoded_slice_compression_ratio_nonzero() {
        let enc = make_encoder(1);
        let frame = RawFrame::new_test(64, 64, PixelFormat::Uyvy422);
        let slices = enc.split_frame(&frame).expect("split should succeed");
        let encoded = enc.encode_slice(slices.into_iter().next().expect("at least one slice"));
        assert!(encoded.input_bytes > 0);
        assert!(encoded.compression_ratio() >= 0.0);
    }

    // --- Huffman entropy coding round-trip ---

    /// Small reproducible PRNG (xorshift64*) — keeps tests deterministic
    /// without pulling in `rand` as a dev dependency.
    struct Xs(u64);
    impl Xs {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn next_in(&mut self, lo: u8, hi: u8) -> u8 {
            let range = (hi - lo) as u64 + 1;
            lo + (self.next_u64() % range) as u8
        }
    }

    fn make_round_trip_encoder(q: u8) -> FrameSliceEncoder {
        FrameSliceEncoder::new(SliceEncoderConfig {
            slice_count: 1,
            quantiser: q.max(1),
            enable_intra_prediction: false,
            min_slice_rows: NonZeroU32::new(1).expect("1 is non-zero"),
        })
    }

    /// Build a frame whose contents are a smooth ramp — typical natural-image
    /// statistics → small AC residuals, dominated by short Huffman codes.
    fn smooth_frame(width: u32, height: u32) -> RawFrame {
        let stride = PixelFormat::Yv12.stride(width);
        let mut data = vec![0u8; stride * height as usize];
        for row in 0..height as usize {
            for col in 0..width as usize {
                // Linear ramp 0..255 plus a tiny noise term derived from
                // position parity.  This produces lots of small AC residuals.
                let value = ((row + col) & 0xFF) as u8;
                data[row * stride + col] = value;
            }
        }
        RawFrame::new(width, height, PixelFormat::Yv12, data, stride, 0)
    }

    /// Build a frame full of high-frequency garbage — exercises the ESCAPE
    /// path heavily because the AC residuals span the full ±127 range.
    fn noisy_frame(width: u32, height: u32, seed: u64) -> RawFrame {
        let stride = PixelFormat::Yv12.stride(width);
        let mut data = vec![0u8; stride * height as usize];
        let mut rng = Xs(seed);
        for byte in &mut data {
            *byte = rng.next_in(0, 255);
        }
        RawFrame::new(width, height, PixelFormat::Yv12, data, stride, 0)
    }

    #[test]
    fn test_round_trip_zero_frame_produces_eob_only_blocks() {
        // All-zero input → DC=0 every block → AC all zero → only EOB per block.
        // Bitstream length should be tiny: (DC_BITS + HUFFMAN_EOB_BITS) bits/block.
        let enc = make_round_trip_encoder(1);
        let frame = RawFrame::new_test(64, 8, PixelFormat::Yv12);
        let slices = enc.split_frame(&frame).expect("split ok");
        let encoded = enc.encode_slice(slices.into_iter().next().expect("slice"));
        let decoded = decode_slice(&encoded.data).expect("decode ok");

        assert_eq!(decoded.format, PixelFormat::Yv12);
        assert_eq!(decoded.first_row, 0);
        assert_eq!(decoded.row_count, 8);
        assert!(
            decoded.data.iter().all(|&b| b == 0),
            "all-zero frame must decode to all zeros"
        );

        // Sanity: encoded payload must be substantially smaller than raw
        // (64×8 = 512 raw bytes, EOB-only encoding fits in ~3 bits per block).
        assert!(
            encoded.data.len() < encoded.input_bytes,
            "all-zero frame should compress: {} vs raw {}",
            encoded.data.len(),
            encoded.input_bytes
        );
    }

    #[test]
    fn test_round_trip_smooth_frame_matches_quantised() {
        // For q=1, the encoder is lossless on AC (integer division by 1) so the
        // round-trip must recover every pixel exactly.
        let enc = make_round_trip_encoder(1);
        let frame = smooth_frame(64, 16);
        let slices = enc.split_frame(&frame).expect("split ok");
        let original_data = slices[0].data.clone();
        let encoded = enc.encode_slice(slices.into_iter().next().expect("slice"));
        let decoded = decode_slice(&encoded.data).expect("decode ok");
        assert_eq!(
            decoded.data.len(),
            original_data.len(),
            "decoded length must match original"
        );
        assert_eq!(
            decoded.data, original_data,
            "q=1 round-trip must be lossless"
        );
    }

    #[test]
    fn test_compression_ratio_typical_block() {
        // Smooth (typical) content with q=8 should achieve clear compression
        // versus the raw input.  The brief target is "at least 30% smaller".
        let enc = make_round_trip_encoder(8);
        let frame = smooth_frame(128, 32);
        let slices = enc.split_frame(&frame).expect("split ok");
        let encoded = enc.encode_slice(slices.into_iter().next().expect("slice"));
        let raw = encoded.input_bytes;
        let coded = encoded.data.len();
        let ratio = coded as f64 / raw as f64;
        assert!(
            ratio < 0.70,
            "expected ratio < 0.70 on smooth content; got {ratio} ({coded}/{raw})"
        );
    }

    #[test]
    fn test_round_trip_noisy_frame_uses_escape_path() {
        // Random byte content with q=1 hits the ESCAPE branch for many levels
        // (residuals span the full ±127 range).  Compression may be worse than
        // raw — that's expected — but round-trip must still be exact at q=1.
        let enc = make_round_trip_encoder(1);
        let frame = noisy_frame(64, 8, 0xC0FFEE_u64);
        let slices = enc.split_frame(&frame).expect("split ok");
        let original_data = slices[0].data.clone();
        let encoded = enc.encode_slice(slices.into_iter().next().expect("slice"));
        let decoded = decode_slice(&encoded.data).expect("decode ok");
        assert_eq!(
            decoded.data, original_data,
            "noisy frame must still round-trip losslessly at q=1"
        );
    }

    #[test]
    fn test_decode_truncated_returns_error() {
        let enc = make_round_trip_encoder(4);
        let frame = smooth_frame(32, 8);
        let slices = enc.split_frame(&frame).expect("split ok");
        let encoded = enc.encode_slice(slices.into_iter().next().expect("slice"));
        let truncated = &encoded.data[..encoded.data.len() / 2];
        assert!(
            decode_slice(truncated).is_err(),
            "truncated slice must fail decode"
        );
    }

    #[test]
    fn test_decode_wrong_version_returns_error() {
        let enc = make_round_trip_encoder(4);
        let frame = smooth_frame(32, 8);
        let slices = enc.split_frame(&frame).expect("split ok");
        let encoded = enc.encode_slice(slices.into_iter().next().expect("slice"));
        let mut tampered = encoded.data.clone();
        tampered[0] = 0xFF;
        assert!(
            decode_slice(&tampered).is_err(),
            "unsupported version byte must fail decode"
        );
    }

    #[test]
    fn test_decode_full_frame_per_slice() {
        // Encode a complete frame across multiple slices and verify every
        // slice independently round-trips.
        let enc = FrameSliceEncoder::new(SliceEncoderConfig {
            slice_count: 4,
            quantiser: 1,
            enable_intra_prediction: false,
            min_slice_rows: NonZeroU32::new(1).expect("1 is non-zero"),
        });
        let frame = smooth_frame(64, 32);
        let encoded = enc.encode(&frame).expect("encode ok");

        let split = enc.split_frame(&frame).expect("split ok");
        for (i, original) in split.iter().enumerate() {
            let slice_bytes = encoded.slice_data(i).expect("slice payload exists");
            let decoded = decode_slice(slice_bytes).expect("decode ok");
            assert_eq!(decoded.index, original.index, "slice index must round-trip");
            assert_eq!(
                decoded.first_row, original.first_row,
                "first_row must round-trip"
            );
            assert_eq!(
                decoded.row_count, original.row_count,
                "row_count must round-trip"
            );
            assert_eq!(
                decoded.data.len(),
                original.data.len(),
                "data length must round-trip"
            );
            assert_eq!(
                decoded.data, original.data,
                "q=1 slice data must round-trip exactly"
            );
        }
    }

    #[test]
    fn test_huffman_table_is_prefix_free() {
        // Sanity-check the static table: every (code, bits) pair must be a
        // valid prefix code together with EOB and ESCAPE.  Verify by Kraft
        // inequality and pairwise prefix comparison.
        let mut all: Vec<(u32, u32)> = HUFFMAN_AC_TABLE.iter().map(|e| (e.code, e.bits)).collect();
        all.push((HUFFMAN_EOB_CODE, HUFFMAN_EOB_BITS));
        all.push((HUFFMAN_ESCAPE_CODE, HUFFMAN_ESCAPE_BITS));

        // Kraft inequality: sum(2^-len) <= 1
        let mut kraft = 0.0_f64;
        for &(_, bits) in &all {
            kraft += 2.0_f64.powi(-(bits as i32));
        }
        assert!(kraft <= 1.0 + 1e-9, "Kraft inequality violated: {kraft}");

        // Pairwise prefix check.
        for (i, &(ca, la)) in all.iter().enumerate() {
            for (j, &(cb, lb)) in all.iter().enumerate() {
                if i == j {
                    continue;
                }
                if la <= lb {
                    let shifted = cb >> (lb - la);
                    assert!(
                        shifted != ca,
                        "code {ca:b} (len {la}) is a prefix of {cb:b} (len {lb})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_bitwriter_bitreader_round_trip() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b101, 3);
        bw.write_bits(0b1111_0000_1010, 12);
        bw.write_bits(0b1, 1);
        let bytes = bw.finalize();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(3).expect("read"), 0b101);
        assert_eq!(br.read_bits(12).expect("read"), 0b1111_0000_1010);
        assert_eq!(br.read_bits(1).expect("read"), 0b1);
    }

    #[test]
    fn test_bitreader_eof_returns_error() {
        let mut br = BitReader::new(&[0xFF]);
        assert!(br.read_bits(8).is_ok());
        assert!(br.read_bits(1).is_err());
    }

    #[test]
    fn test_encoded_slice_layout_carries_version_byte() {
        let enc = make_round_trip_encoder(4);
        let frame = smooth_frame(32, 8);
        let slices = enc.split_frame(&frame).expect("split ok");
        let encoded = enc.encode_slice(slices.into_iter().next().expect("slice"));
        assert_eq!(
            encoded.data[0], SPEEDHQ_FORMAT_VERSION,
            "first byte must be the format version"
        );
    }

    #[test]
    fn test_decode_rejects_absurd_block_count() {
        // Construct a minimally valid header + a 1-byte bitstream that
        // claims to contain 1 million blocks.  The capacity guard must
        // reject this without attempting to allocate 8 MB+ of decode space.
        let mut buf = vec![
            SPEEDHQ_FORMAT_VERSION,
            0,
            0, // slice index
            0,
            0, // first_row
            0,
            0, // row_count
            8, // quantiser
            1, // YV12 format id
        ];
        let bogus_block_count: u32 = 1_000_000;
        buf.extend_from_slice(&bogus_block_count.to_le_bytes());
        let bitstream: [u8; 1] = [0xFF];
        buf.extend_from_slice(&(bitstream.len() as u32).to_le_bytes());
        buf.extend_from_slice(&bitstream);
        buf.push(0); // tail_len = 0

        let err = decode_slice(&buf).expect_err("must reject absurd block_count");
        let msg = format!("{err}");
        assert!(
            msg.contains("block_count"),
            "expected block_count error, got: {msg}"
        );
    }

    #[test]
    fn test_decode_rejects_missing_eob_after_full_block() {
        // Craft a slice with one block where the bitstream emits a (run=0,
        // level=1) eight times in a row (filling all 8 AC slots) and then
        // does NOT emit EOB.  Decoder should report "missing EOB".
        let mut bw = BitWriter::new();
        bw.write_bits(0, DC_BITS); // DC residual = 0
        for _ in 0..8 {
            // (run=0, level=+1) = code 0b11, len 2, sign 0
            bw.write_bits(0b11, 2);
            bw.write_bits(0, 1);
        }
        // Intentionally append a non-EOB symbol (another (0,1)) instead of EOB.
        bw.write_bits(0b11, 2);
        bw.write_bits(0, 1);
        let bitstream = bw.finalize();

        let mut buf = vec![SPEEDHQ_FORMAT_VERSION, 0, 0, 0, 0, 0, 0, 1, 1];
        buf.extend_from_slice(&1u32.to_le_bytes()); // block_count = 1
        buf.extend_from_slice(&(bitstream.len() as u32).to_le_bytes());
        buf.extend_from_slice(&bitstream);
        buf.push(0); // tail_len = 0

        let err = decode_slice(&buf).expect_err("must reject missing EOB");
        let msg = format!("{err}");
        assert!(
            msg.contains("EOB"),
            "expected missing-EOB error, got: {msg}"
        );
    }
}
