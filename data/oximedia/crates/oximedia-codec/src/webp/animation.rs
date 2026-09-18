//! WebP animation encoding and decoding.
//!
//! Implements the WebP animation container format (Extended WebP with ANIM/ANMF chunks).
//!
//! # Container structure
//!
//! ```text
//! RIFF <file_size>
//!   WEBP
//!   VP8X <10 bytes>   (has_animation=1, canvas_width-1, canvas_height-1)
//!   ANIM <6 bytes>    (background_color BGRA u32 LE, loop_count u16 LE)
//!   ANMF <frame_data> (per-frame: x/2, y/2, w-1, h-1, duration 24-bit, flags, VP8L bitstream)
//!   ...
//! ```
//!
//! Each ANMF chunk embeds a VP8L lossless bitstream for the frame pixels.

use crate::error::{CodecError, CodecResult};
use crate::webp::vp8l_encoder::Vp8lEncoder;

// ── Constants ──────────────────────────────────────────────────────────────────

const RIFF_MAGIC: &[u8; 4] = b"RIFF";
const WEBP_MAGIC: &[u8; 4] = b"WEBP";

const FOURCC_VP8X: [u8; 4] = *b"VP8X";
const FOURCC_ANIM: [u8; 4] = *b"ANIM";
const FOURCC_ANMF: [u8; 4] = *b"ANMF";
const FOURCC_VP8L: [u8; 4] = *b"VP8L";

/// VP8X flag bit for animation.
const VP8X_FLAG_ANIMATION: u8 = 1 << 1;
/// VP8X flag bit for alpha.
const VP8X_FLAG_ALPHA: u8 = 1 << 4;

/// Minimum bytes needed for RIFF header (RIFF tag + file size + WEBP).
const RIFF_HEADER_SIZE: usize = 12;
/// Size of each chunk header (FourCC + u32 size).
const CHUNK_HEADER_SIZE: usize = 8;
/// ANMF chunk header payload size before the bitstream data.
/// X/2 (3 bytes) + Y/2 (3 bytes) + (W-1) (3 bytes) + (H-1) (3 bytes) + duration (3 bytes) + flags (1 byte) = 16 bytes.
const ANMF_HEADER_SIZE: usize = 16;
/// ANIM chunk payload size: background_color (4) + loop_count (2).
const ANIM_PAYLOAD_SIZE: usize = 6;
/// VP8X chunk payload size.
const VP8X_PAYLOAD_SIZE: usize = 10;

// ── Public types ───────────────────────────────────────────────────────────────

/// Configuration for an animated WebP sequence.
#[derive(Debug, Clone)]
pub struct WebpAnimConfig {
    /// Number of times to loop the animation. 0 = infinite.
    pub loop_count: u16,
    /// Canvas background color as 0xAARRGGBB (stored as BGRA in the file).
    pub background_color: u32,
}

impl Default for WebpAnimConfig {
    fn default() -> Self {
        Self {
            loop_count: 0,
            background_color: 0xFF000000, // opaque black
        }
    }
}

/// A single frame within an animated WebP.
#[derive(Debug, Clone)]
pub struct WebpAnimFrame {
    /// Raw RGBA pixel data (4 bytes per pixel, row-major).
    pub pixels: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame display timestamp in milliseconds.
    pub timestamp_ms: u32,
    /// X offset on the canvas (must be divisible by 2).
    pub x_offset: u32,
    /// Y offset on the canvas (must be divisible by 2).
    pub y_offset: u32,
    /// Whether to alpha-blend this frame over the previous one.
    pub blend: bool,
    /// Whether to dispose (clear to background) the frame area after display.
    pub dispose: bool,
}

impl WebpAnimFrame {
    /// Validate frame constraints required by the WebP spec.
    fn validate(&self) -> CodecResult<()> {
        if self.x_offset % 2 != 0 {
            return Err(CodecError::InvalidParameter(format!(
                "x_offset {} must be divisible by 2",
                self.x_offset
            )));
        }
        if self.y_offset % 2 != 0 {
            return Err(CodecError::InvalidParameter(format!(
                "y_offset {} must be divisible by 2",
                self.y_offset
            )));
        }
        if self.width == 0 || self.height == 0 {
            return Err(CodecError::InvalidParameter(
                "Frame dimensions must be non-zero".into(),
            ));
        }
        let expected = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|px| px.checked_mul(4))
            .ok_or_else(|| {
                CodecError::InvalidParameter("Frame pixel buffer size overflow".into())
            })?;
        if self.pixels.len() != expected {
            return Err(CodecError::InvalidParameter(format!(
                "pixels length {} does not match {}×{}×4 = {}",
                self.pixels.len(),
                self.width,
                self.height,
                expected
            )));
        }
        Ok(())
    }
}

// ── Encoder ────────────────────────────────────────────────────────────────────

/// Encoder for animated WebP files.
pub struct WebpAnimEncoder;

impl WebpAnimEncoder {
    /// Encode a sequence of frames into an animated WebP byte stream.
    ///
    /// The canvas dimensions are derived from the maximum extent of all frames
    /// (x_offset + width, y_offset + height). All frames must have valid pixel
    /// data matching their declared width/height.
    pub fn encode(frames: &[WebpAnimFrame], config: &WebpAnimConfig) -> CodecResult<Vec<u8>> {
        if frames.is_empty() {
            return Err(CodecError::InvalidParameter(
                "Animation must contain at least one frame".into(),
            ));
        }

        // Validate all frames up front.
        for (i, frame) in frames.iter().enumerate() {
            frame
                .validate()
                .map_err(|e| CodecError::InvalidParameter(format!("Frame {i}: {e}")))?;
        }

        // Compute canvas dimensions.
        let canvas_width = frames
            .iter()
            .map(|f| f.x_offset + f.width)
            .max()
            .unwrap_or(1);
        let canvas_height = frames
            .iter()
            .map(|f| f.y_offset + f.height)
            .max()
            .unwrap_or(1);

        // Detect whether any frame has a non-trivial alpha channel.
        let has_alpha = frames.iter().any(|f| has_non_opaque_alpha(&f.pixels));

        // Build VP8X chunk payload.
        let vp8x_payload = encode_vp8x(canvas_width, canvas_height, true, has_alpha);

        // Build ANIM chunk payload.
        let anim_payload = encode_anim_chunk(config);

        // Build ANMF chunks for each frame.
        let anmf_chunks: Vec<Vec<u8>> = frames
            .iter()
            .enumerate()
            .map(|(i, frame)| encode_anmf_chunk(frame, i))
            .collect::<CodecResult<_>>()?;

        // Compute total RIFF body size.
        // RIFF body = "WEBP" (4) + VP8X chunk + ANIM chunk + all ANMF chunks
        let mut body_size: usize = 4; // "WEBP"
        body_size += chunk_wire_size(VP8X_PAYLOAD_SIZE);
        body_size += chunk_wire_size(ANIM_PAYLOAD_SIZE);
        for anmf in &anmf_chunks {
            body_size += chunk_wire_size(anmf.len());
        }

        let mut out = Vec::with_capacity(RIFF_HEADER_SIZE + body_size);

        // RIFF header.
        out.extend_from_slice(RIFF_MAGIC);
        write_u32_le(&mut out, body_size as u32);
        out.extend_from_slice(WEBP_MAGIC);

        // VP8X chunk.
        write_chunk(&mut out, &FOURCC_VP8X, &vp8x_payload);

        // ANIM chunk.
        write_chunk(&mut out, &FOURCC_ANIM, &anim_payload);

        // ANMF chunks.
        for anmf in &anmf_chunks {
            write_chunk(&mut out, &FOURCC_ANMF, anmf);
        }

        Ok(out)
    }
}

// ── Decoder ────────────────────────────────────────────────────────────────────

/// Decoder for animated WebP files.
pub struct WebpAnimDecoder;

impl WebpAnimDecoder {
    /// Decode an animated WebP byte stream into frames and configuration.
    ///
    /// Returns the decoded frames (with RGBA pixel data) and animation config.
    pub fn decode(data: &[u8]) -> CodecResult<(Vec<WebpAnimFrame>, WebpAnimConfig)> {
        validate_riff_header(data)?;

        let chunks = parse_chunks(&data[RIFF_HEADER_SIZE..], data.len() - RIFF_HEADER_SIZE)?;

        // Find the ANIM chunk.
        let anim_payload = chunks
            .iter()
            .find(|(cc, _)| cc == &FOURCC_ANIM)
            .map(|(_, d)| d.as_slice())
            .ok_or_else(|| CodecError::InvalidBitstream("Missing ANIM chunk".into()))?;

        let config = decode_anim_chunk(anim_payload)?;

        // Decode each ANMF chunk.
        let frames: Vec<WebpAnimFrame> = chunks
            .iter()
            .filter(|(cc, _)| cc == &FOURCC_ANMF)
            .map(|(_, d)| decode_anmf_chunk(d))
            .collect::<CodecResult<_>>()?;

        if frames.is_empty() {
            return Err(CodecError::InvalidBitstream(
                "Animated WebP contains no ANMF frames".into(),
            ));
        }

        Ok((frames, config))
    }

    /// Return the number of animation frames without fully decoding all pixel data.
    pub fn frame_count(data: &[u8]) -> CodecResult<u32> {
        if !Self::is_webp_anim(data) {
            return Err(CodecError::InvalidBitstream(
                "Data is not an animated WebP".into(),
            ));
        }
        let chunks = parse_chunks(&data[RIFF_HEADER_SIZE..], data.len() - RIFF_HEADER_SIZE)?;
        let count = chunks.iter().filter(|(cc, _)| cc == &FOURCC_ANMF).count();
        Ok(count as u32)
    }

    /// Return true if the byte slice is a valid animated WebP (RIFF+WEBP with ANIM chunk).
    pub fn is_webp_anim(data: &[u8]) -> bool {
        if data.len() < RIFF_HEADER_SIZE {
            return false;
        }
        if &data[0..4] != RIFF_MAGIC || &data[8..12] != WEBP_MAGIC {
            return false;
        }
        // Quick scan for ANIM chunk FourCC without full parse.
        let body = &data[RIFF_HEADER_SIZE..];
        has_chunk_fourcc(body, &FOURCC_ANIM)
    }
}

// ── Encoding helpers ───────────────────────────────────────────────────────────

/// Build the VP8X chunk payload (10 bytes) for animated WebP.
fn encode_vp8x(
    canvas_width: u32,
    canvas_height: u32,
    has_anim: bool,
    has_alpha: bool,
) -> [u8; VP8X_PAYLOAD_SIZE] {
    let mut buf = [0u8; VP8X_PAYLOAD_SIZE];
    let mut flags: u8 = 0;
    if has_anim {
        flags |= VP8X_FLAG_ANIMATION;
    }
    if has_alpha {
        flags |= VP8X_FLAG_ALPHA;
    }
    buf[0] = flags;
    // bytes 1..4 reserved (zero)
    let w = canvas_width.saturating_sub(1);
    buf[4] = (w & 0xFF) as u8;
    buf[5] = ((w >> 8) & 0xFF) as u8;
    buf[6] = ((w >> 16) & 0xFF) as u8;
    let h = canvas_height.saturating_sub(1);
    buf[7] = (h & 0xFF) as u8;
    buf[8] = ((h >> 8) & 0xFF) as u8;
    buf[9] = ((h >> 16) & 0xFF) as u8;
    buf
}

/// Build the ANIM chunk payload (6 bytes).
///
/// background_color is stored as BGRA (Blue, Green, Red, Alpha) per the spec.
fn encode_anim_chunk(config: &WebpAnimConfig) -> [u8; ANIM_PAYLOAD_SIZE] {
    let mut buf = [0u8; ANIM_PAYLOAD_SIZE];
    // background_color: AARRGGBB -> stored as BGRA LE
    let aa = ((config.background_color >> 24) & 0xFF) as u8;
    let rr = ((config.background_color >> 16) & 0xFF) as u8;
    let gg = ((config.background_color >> 8) & 0xFF) as u8;
    let bb = (config.background_color & 0xFF) as u8;
    buf[0] = bb;
    buf[1] = gg;
    buf[2] = rr;
    buf[3] = aa;
    let lc = config.loop_count.to_le_bytes();
    buf[4] = lc[0];
    buf[5] = lc[1];
    buf
}

/// Build the ANMF chunk payload for a single frame (header + VP8L bitstream).
///
/// ANMF payload layout (all LE):
/// - X/2 (24-bit): frame x offset / 2
/// - Y/2 (24-bit): frame y offset / 2
/// - (W-1) (24-bit): frame width - 1
/// - (H-1) (24-bit): frame height - 1
/// - Duration (24-bit): display duration in ms
/// - Flags (8-bit): bit 1 = dispose method, bit 2 = blend method
/// - Frame data (VP8L chunk: FourCC "VP8L" + LE u32 size + bitstream)
fn encode_anmf_chunk(frame: &WebpAnimFrame, _index: usize) -> CodecResult<Vec<u8>> {
    // Encode pixels as VP8L.
    let vp8l_data = encode_frame_vp8l(frame)?;

    // ANMF frame data = VP8L chunk (FourCC + size + bitstream).
    // We embed the VP8L as a sub-chunk within the ANMF payload.
    let inner_chunk_size =
        CHUNK_HEADER_SIZE + vp8l_data.len() + if vp8l_data.len() % 2 != 0 { 1 } else { 0 };
    let mut payload = Vec::with_capacity(ANMF_HEADER_SIZE + inner_chunk_size);

    // Offsets (24-bit LE, divided by 2).
    let x2 = frame.x_offset / 2;
    let y2 = frame.y_offset / 2;
    write_u24_le(&mut payload, x2);
    write_u24_le(&mut payload, y2);

    // Dimensions (24-bit LE, minus 1).
    write_u24_le(&mut payload, frame.width.saturating_sub(1));
    write_u24_le(&mut payload, frame.height.saturating_sub(1));

    // Duration (24-bit LE, milliseconds).
    write_u24_le(&mut payload, frame.timestamp_ms.min(0x00FF_FFFF));

    // Flags byte.
    // bit 0: dispose method (0 = do not dispose, 1 = dispose to background)
    // bit 1: blending method (0 = use alpha blending, 1 = do not blend)
    let mut flags: u8 = 0;
    if frame.dispose {
        flags |= 0x01;
    }
    if !frame.blend {
        flags |= 0x02;
    }
    payload.push(flags);

    // Embed VP8L as a sub-chunk.
    write_chunk(&mut payload, &FOURCC_VP8L, &vp8l_data);

    Ok(payload)
}

/// Convert RGBA pixels to ARGB u32 values for the VP8L encoder.
fn rgba_to_argb_u32(pixels: &[u8], width: u32, height: u32) -> CodecResult<Vec<u32>> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| CodecError::InvalidParameter("Pixel buffer size overflow".into()))?;
    if pixels.len() < expected {
        return Err(CodecError::InvalidParameter(format!(
            "Pixel buffer too small: need {expected}, have {}",
            pixels.len()
        )));
    }
    let count = (width as usize) * (height as usize);
    let mut argb = Vec::with_capacity(count);
    for i in 0..count {
        let r = pixels[i * 4] as u32;
        let g = pixels[i * 4 + 1] as u32;
        let b = pixels[i * 4 + 2] as u32;
        let a = pixels[i * 4 + 3] as u32;
        argb.push((a << 24) | (r << 16) | (g << 8) | b);
    }
    Ok(argb)
}

/// Encode a single animation frame to VP8L bitstream bytes.
fn encode_frame_vp8l(frame: &WebpAnimFrame) -> CodecResult<Vec<u8>> {
    let argb = rgba_to_argb_u32(&frame.pixels, frame.width, frame.height)?;
    let has_alpha = has_non_opaque_alpha(&frame.pixels);
    let encoder = Vp8lEncoder::new(0);
    encoder.encode(&argb, frame.width, frame.height, has_alpha)
}

/// Return true if any pixel in the RGBA buffer has alpha < 255.
fn has_non_opaque_alpha(pixels: &[u8]) -> bool {
    pixels.chunks_exact(4).any(|px| px[3] < 255)
}

// ── Decoding helpers ───────────────────────────────────────────────────────────

/// Validate the RIFF/WEBP header magic bytes.
fn validate_riff_header(data: &[u8]) -> CodecResult<()> {
    if data.len() < RIFF_HEADER_SIZE {
        return Err(CodecError::InvalidBitstream(
            "Data too small for RIFF header".into(),
        ));
    }
    if &data[0..4] != RIFF_MAGIC {
        return Err(CodecError::InvalidBitstream(
            "Missing RIFF magic bytes".into(),
        ));
    }
    if &data[8..12] != WEBP_MAGIC {
        return Err(CodecError::InvalidBitstream(
            "Missing WEBP form type magic".into(),
        ));
    }
    Ok(())
}

/// Parse all top-level RIFF chunks from the WEBP body.
///
/// `body` is `data[RIFF_HEADER_SIZE..]`, i.e. everything after the 12-byte
/// RIFF+size+WEBP header.  The chunk stream starts immediately at offset 0.
///
/// Returns a list of (fourcc, payload_bytes) pairs.
fn parse_chunks(body: &[u8], _body_len: usize) -> CodecResult<Vec<([u8; 4], Vec<u8>)>> {
    let mut offset = 0usize;
    let mut chunks = Vec::new();

    while offset + CHUNK_HEADER_SIZE <= body.len() {
        let mut fourcc = [0u8; 4];
        fourcc.copy_from_slice(&body[offset..offset + 4]);
        let chunk_size = read_u32_le(&body[offset + 4..offset + 8]) as usize;
        offset += CHUNK_HEADER_SIZE;

        if offset + chunk_size > body.len() {
            return Err(CodecError::InvalidBitstream(format!(
                "Chunk '{}' at offset {} declares size {} but only {} bytes remain",
                String::from_utf8_lossy(&fourcc),
                offset - CHUNK_HEADER_SIZE,
                chunk_size,
                body.len().saturating_sub(offset),
            )));
        }

        let payload = body[offset..offset + chunk_size].to_vec();
        chunks.push((fourcc, payload));

        offset += chunk_size;
        if chunk_size % 2 != 0 {
            offset += 1; // skip pad byte
        }
    }

    Ok(chunks)
}

/// Decode the ANIM chunk payload into a WebpAnimConfig.
fn decode_anim_chunk(data: &[u8]) -> CodecResult<WebpAnimConfig> {
    if data.len() < ANIM_PAYLOAD_SIZE {
        return Err(CodecError::InvalidBitstream(format!(
            "ANIM chunk too small: need {ANIM_PAYLOAD_SIZE}, got {}",
            data.len()
        )));
    }
    // Stored as BGRA LE.
    let bb = data[0] as u32;
    let gg = data[1] as u32;
    let rr = data[2] as u32;
    let aa = data[3] as u32;
    let background_color = (aa << 24) | (rr << 16) | (gg << 8) | bb;
    let loop_count = u16::from_le_bytes([data[4], data[5]]);
    Ok(WebpAnimConfig {
        loop_count,
        background_color,
    })
}

/// Decode an ANMF chunk payload into a WebpAnimFrame with decoded pixels.
fn decode_anmf_chunk(data: &[u8]) -> CodecResult<WebpAnimFrame> {
    if data.len() < ANMF_HEADER_SIZE {
        return Err(CodecError::InvalidBitstream(format!(
            "ANMF chunk too small: need {ANMF_HEADER_SIZE} bytes for header, got {}",
            data.len()
        )));
    }

    let x_offset = read_u24_le(&data[0..3]) * 2;
    let y_offset = read_u24_le(&data[3..6]) * 2;
    let width = read_u24_le(&data[6..9]) + 1;
    let height = read_u24_le(&data[9..12]) + 1;
    let timestamp_ms = read_u24_le(&data[12..15]);
    let flags = data[15];

    let dispose = (flags & 0x01) != 0;
    let blend = (flags & 0x02) == 0;

    // The remaining bytes should be a VP8L sub-chunk.
    let frame_data = &data[ANMF_HEADER_SIZE..];
    let pixels = decode_vp8l_subchunk(frame_data, width, height)?;

    Ok(WebpAnimFrame {
        pixels,
        width,
        height,
        timestamp_ms,
        x_offset,
        y_offset,
        blend,
        dispose,
    })
}

/// Decode pixels from a VP8L sub-chunk embedded in an ANMF payload.
fn decode_vp8l_subchunk(data: &[u8], width: u32, height: u32) -> CodecResult<Vec<u8>> {
    // Sub-chunk: FourCC(4) + size(4) + VP8L bitstream.
    if data.len() < CHUNK_HEADER_SIZE {
        return Err(CodecError::InvalidBitstream(
            "ANMF frame data too small for sub-chunk header".into(),
        ));
    }
    let fourcc = &data[0..4];
    if fourcc != FOURCC_VP8L {
        return Err(CodecError::InvalidBitstream(format!(
            "Expected VP8L sub-chunk in ANMF, got '{}'",
            String::from_utf8_lossy(fourcc)
        )));
    }
    let chunk_size = read_u32_le(&data[4..8]) as usize;
    if data.len() < CHUNK_HEADER_SIZE + chunk_size {
        return Err(CodecError::InvalidBitstream(
            "VP8L sub-chunk data truncated".into(),
        ));
    }
    let vp8l_data = &data[CHUNK_HEADER_SIZE..CHUNK_HEADER_SIZE + chunk_size];
    decode_vp8l_to_rgba(vp8l_data, width, height)
}

/// Decode a VP8L bitstream to RGBA pixels using the existing Vp8lDecoder.
fn decode_vp8l_to_rgba(vp8l_data: &[u8], width: u32, height: u32) -> CodecResult<Vec<u8>> {
    use crate::webp::vp8l_decoder::Vp8lDecoder;

    let decoded = Vp8lDecoder::new()
        .decode(vp8l_data)
        .map_err(|e| CodecError::DecoderError(format!("VP8L decode failed: {e}")))?;

    // Defend against an ANMF header that declares different dimensions than the
    // embedded VP8L image. Downstream consumers index the returned buffer using
    // the ANMF-declared width/height, so a mismatch (declared larger than the
    // decoded image) would read out of bounds. Require an exact match.
    let expected = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| CodecError::InvalidBitstream("ANMF frame dimensions overflow".into()))?;
    if decoded.pixels.len() != expected {
        return Err(CodecError::InvalidBitstream(format!(
            "ANMF declares {width}x{height} but embedded VP8L decoded {} pixels",
            decoded.pixels.len()
        )));
    }

    // decoded.pixels is Vec<u32> in ARGB order; convert to RGBA bytes.
    let mut rgba = Vec::with_capacity(decoded.pixels.len() * 4);
    for argb in &decoded.pixels {
        let a = (argb >> 24) as u8;
        let r = (argb >> 16) as u8;
        let g = (argb >> 8) as u8;
        let b = *argb as u8;
        rgba.push(r);
        rgba.push(g);
        rgba.push(b);
        rgba.push(a);
    }
    Ok(rgba)
}

/// Quick scan of chunk stream bytes for a given FourCC without full parsing.
///
/// `body` is the bytes immediately after the 12-byte RIFF+size+WEBP header,
/// so the chunk stream starts at offset 0.
fn has_chunk_fourcc(body: &[u8], target: &[u8; 4]) -> bool {
    let mut offset = 0usize;
    while offset + CHUNK_HEADER_SIZE <= body.len() {
        let fourcc = &body[offset..offset + 4];
        if fourcc == target.as_ref() {
            return true;
        }
        let chunk_size = read_u32_le(&body[offset + 4..offset + 8]) as usize;
        offset += CHUNK_HEADER_SIZE + chunk_size;
        if chunk_size % 2 != 0 {
            offset += 1;
        }
    }
    false
}

// ── Wire format helpers ────────────────────────────────────────────────────────

/// Write a RIFF chunk: FourCC + LE u32 size + payload + optional pad byte.
fn write_chunk(buf: &mut Vec<u8>, fourcc: &[u8; 4], data: &[u8]) {
    buf.extend_from_slice(fourcc);
    write_u32_le(buf, data.len() as u32);
    buf.extend_from_slice(data);
    if data.len() % 2 != 0 {
        buf.push(0);
    }
}

/// Return the wire size of a chunk (header + payload + optional pad byte).
fn chunk_wire_size(payload_len: usize) -> usize {
    CHUNK_HEADER_SIZE + payload_len + (payload_len % 2)
}

fn write_u32_le(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u24_le(buf: &mut Vec<u8>, v: u32) {
    buf.push((v & 0xFF) as u8);
    buf.push(((v >> 8) & 0xFF) as u8);
    buf.push(((v >> 16) & 0xFF) as u8);
}

fn read_u32_le(data: &[u8]) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&data[..4]);
    u32::from_le_bytes(b)
}

fn read_u24_le(data: &[u8]) -> u32 {
    u32::from(data[0]) | (u32::from(data[1]) << 8) | (u32::from(data[2]) << 16)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a solid-color RGBA frame.
    fn make_solid_frame(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> WebpAnimFrame {
        let pixels = (0..width * height)
            .flat_map(|_| [r, g, b, a])
            .collect::<Vec<u8>>();
        WebpAnimFrame {
            pixels,
            width,
            height,
            timestamp_ms: 0,
            x_offset: 0,
            y_offset: 0,
            blend: true,
            dispose: false,
        }
    }

    /// Build a sequence of timed frames with distinct colours.
    fn make_colour_frames() -> Vec<WebpAnimFrame> {
        let colours: &[(u8, u8, u8, u8, u32)] = &[
            (255, 0, 0, 255, 0),
            (0, 255, 0, 255, 100),
            (0, 0, 255, 255, 200),
        ];
        colours
            .iter()
            .map(|&(r, g, b, a, ts)| {
                let mut frame = make_solid_frame(4, 4, r, g, b, a);
                frame.timestamp_ms = ts;
                frame
            })
            .collect()
    }

    // ── is_webp_anim ──────────────────────────────────────────────────

    #[test]
    fn test_is_webp_anim_true_after_encode() {
        let frames = make_colour_frames();
        let config = WebpAnimConfig::default();
        let data = WebpAnimEncoder::encode(&frames, &config).expect("encode");
        assert!(WebpAnimDecoder::is_webp_anim(&data));
    }

    #[test]
    fn test_is_webp_anim_false_for_empty() {
        assert!(!WebpAnimDecoder::is_webp_anim(&[]));
    }

    #[test]
    fn test_is_webp_anim_false_for_garbage() {
        let junk = vec![0xFFu8; 64];
        assert!(!WebpAnimDecoder::is_webp_anim(&junk));
    }

    #[test]
    fn test_is_webp_anim_false_for_truncated_riff() {
        // Valid RIFF+WEBP magic but truncated — no ANIM chunk.
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(RIFF_MAGIC);
        data[8..12].copy_from_slice(WEBP_MAGIC);
        assert!(!WebpAnimDecoder::is_webp_anim(&data));
    }

    // ── frame_count ───────────────────────────────────────────────────

    #[test]
    fn test_frame_count_single() {
        let frames = vec![make_solid_frame(2, 2, 128, 128, 128, 255)];
        let config = WebpAnimConfig::default();
        let data = WebpAnimEncoder::encode(&frames, &config).expect("encode");
        let count = WebpAnimDecoder::frame_count(&data).expect("count");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_frame_count_multiple() {
        let frames = make_colour_frames();
        let config = WebpAnimConfig::default();
        let data = WebpAnimEncoder::encode(&frames, &config).expect("encode");
        let count = WebpAnimDecoder::frame_count(&data).expect("count");
        assert_eq!(count, 3);
    }

    #[test]
    fn test_frame_count_error_on_non_anim() {
        let data = b"RIFF\x00\x00\x00\x00WEBPnothing-here-at-all";
        assert!(WebpAnimDecoder::frame_count(data).is_err());
    }

    // ── encode / decode roundtrip ─────────────────────────────────────

    #[test]
    fn test_roundtrip_single_frame() {
        let frame = make_solid_frame(4, 4, 200, 100, 50, 255);
        let config = WebpAnimConfig {
            loop_count: 3,
            background_color: 0xFF_FF0000,
        };
        let data = WebpAnimEncoder::encode(&[frame.clone()], &config).expect("encode");
        let (decoded_frames, decoded_config) = WebpAnimDecoder::decode(&data).expect("decode");

        assert_eq!(decoded_config.loop_count, 3);
        assert_eq!(decoded_config.background_color, 0xFF_FF0000);
        assert_eq!(decoded_frames.len(), 1);

        let df = &decoded_frames[0];
        assert_eq!(df.width, 4);
        assert_eq!(df.height, 4);
        assert_eq!(df.timestamp_ms, 0);
        assert_eq!(df.x_offset, 0);
        assert_eq!(df.y_offset, 0);
        assert_eq!(df.pixels.len(), 4 * 4 * 4);
    }

    #[test]
    fn test_roundtrip_multiple_frames() {
        let frames = make_colour_frames();
        let config = WebpAnimConfig {
            loop_count: 0,
            background_color: 0xFF_000000,
        };
        let data = WebpAnimEncoder::encode(&frames, &config).expect("encode");
        let (decoded_frames, decoded_config) = WebpAnimDecoder::decode(&data).expect("decode");

        assert_eq!(decoded_config.loop_count, 0);
        assert_eq!(decoded_frames.len(), 3);

        for (orig, decoded) in frames.iter().zip(decoded_frames.iter()) {
            assert_eq!(decoded.width, orig.width);
            assert_eq!(decoded.height, orig.height);
            assert_eq!(decoded.timestamp_ms, orig.timestamp_ms);
            assert_eq!(decoded.x_offset, orig.x_offset);
            assert_eq!(decoded.y_offset, orig.y_offset);
            assert_eq!(decoded.blend, orig.blend);
            assert_eq!(decoded.dispose, orig.dispose);
            assert_eq!(decoded.pixels.len(), orig.pixels.len());
        }
    }

    #[test]
    fn test_roundtrip_with_alpha() {
        let frame = make_solid_frame(8, 8, 100, 150, 200, 128);
        let config = WebpAnimConfig::default();
        let data = WebpAnimEncoder::encode(&[frame], &config).expect("encode");
        let (decoded_frames, _) = WebpAnimDecoder::decode(&data).expect("decode");
        assert_eq!(decoded_frames.len(), 1);
        assert_eq!(decoded_frames[0].pixels.len(), 8 * 8 * 4);
    }

    #[test]
    fn test_roundtrip_dispose_and_blend_flags() {
        let mut frame = make_solid_frame(4, 4, 0, 0, 0, 255);
        frame.dispose = true;
        frame.blend = false;
        let config = WebpAnimConfig::default();
        let data = WebpAnimEncoder::encode(&[frame], &config).expect("encode");
        let (decoded_frames, _) = WebpAnimDecoder::decode(&data).expect("decode");
        assert_eq!(decoded_frames[0].dispose, true);
        assert_eq!(decoded_frames[0].blend, false);
    }

    #[test]
    fn test_roundtrip_offsets() {
        let mut frame = make_solid_frame(4, 4, 0, 255, 0, 255);
        frame.x_offset = 4;
        frame.y_offset = 6;
        let config = WebpAnimConfig::default();
        let data = WebpAnimEncoder::encode(&[frame], &config).expect("encode");
        let (decoded_frames, _) = WebpAnimDecoder::decode(&data).expect("decode");
        assert_eq!(decoded_frames[0].x_offset, 4);
        assert_eq!(decoded_frames[0].y_offset, 6);
    }

    // ── validation errors ─────────────────────────────────────────────

    #[test]
    fn test_encode_empty_frames_error() {
        let config = WebpAnimConfig::default();
        let result = WebpAnimEncoder::encode(&[], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_odd_x_offset_error() {
        let mut frame = make_solid_frame(4, 4, 0, 0, 0, 255);
        frame.x_offset = 3;
        let config = WebpAnimConfig::default();
        assert!(WebpAnimEncoder::encode(&[frame], &config).is_err());
    }

    #[test]
    fn test_encode_odd_y_offset_error() {
        let mut frame = make_solid_frame(4, 4, 0, 0, 0, 255);
        frame.y_offset = 1;
        let config = WebpAnimConfig::default();
        assert!(WebpAnimEncoder::encode(&[frame], &config).is_err());
    }

    #[test]
    fn test_encode_zero_dimension_error() {
        let frame = WebpAnimFrame {
            pixels: vec![],
            width: 0,
            height: 4,
            timestamp_ms: 0,
            x_offset: 0,
            y_offset: 0,
            blend: true,
            dispose: false,
        };
        let config = WebpAnimConfig::default();
        assert!(WebpAnimEncoder::encode(&[frame], &config).is_err());
    }

    #[test]
    fn test_encode_wrong_pixel_length_error() {
        let frame = WebpAnimFrame {
            pixels: vec![0u8; 10], // wrong: should be 4*4*4=64
            width: 4,
            height: 4,
            timestamp_ms: 0,
            x_offset: 0,
            y_offset: 0,
            blend: true,
            dispose: false,
        };
        let config = WebpAnimConfig::default();
        assert!(WebpAnimEncoder::encode(&[frame], &config).is_err());
    }

    // ── decode error cases ────────────────────────────────────────────

    #[test]
    fn test_decode_too_short() {
        assert!(WebpAnimDecoder::decode(&[0u8; 4]).is_err());
    }

    #[test]
    fn test_decode_bad_magic() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"RIFT"); // wrong magic
        assert!(WebpAnimDecoder::decode(&data).is_err());
    }

    #[test]
    fn test_canvas_dimensions_from_multiple_frames() {
        // Frame 1: 4×4 at (0,0), Frame 2: 4×4 at (4,4) — canvas should be 8×8.
        let mut f1 = make_solid_frame(4, 4, 255, 0, 0, 255);
        f1.x_offset = 0;
        f1.y_offset = 0;
        let mut f2 = make_solid_frame(4, 4, 0, 255, 0, 255);
        f2.x_offset = 4;
        f2.y_offset = 4;

        let config = WebpAnimConfig::default();
        let data = WebpAnimEncoder::encode(&[f1, f2], &config).expect("encode");

        // Verify the VP8X canvas dims embedded in the file.
        // VP8X chunk starts at offset 12 (RIFF header) + 8 (chunk header) = 20.
        // canvas_width-1 is at bytes 4..7 of the VP8X payload (file offset 24).
        let payload_offset = RIFF_HEADER_SIZE + CHUNK_HEADER_SIZE;
        let w = u32::from(data[payload_offset + 4])
            | (u32::from(data[payload_offset + 5]) << 8)
            | (u32::from(data[payload_offset + 6]) << 16);
        let h = u32::from(data[payload_offset + 7])
            | (u32::from(data[payload_offset + 8]) << 8)
            | (u32::from(data[payload_offset + 9]) << 16);
        assert_eq!(w + 1, 8); // canvas_width = 8
        assert_eq!(h + 1, 8); // canvas_height = 8

        let count = WebpAnimDecoder::frame_count(&data).expect("count");
        assert_eq!(count, 2);
    }

    #[test]
    fn test_pixel_fidelity_solid_colour() {
        // Solid green 2×2 — lossless encode/decode should round-trip perfectly.
        let frame = make_solid_frame(2, 2, 0, 255, 0, 255);
        let config = WebpAnimConfig::default();
        let data = WebpAnimEncoder::encode(&[frame.clone()], &config).expect("encode");
        let (decoded, _) = WebpAnimDecoder::decode(&data).expect("decode");
        assert_eq!(decoded[0].pixels, frame.pixels);
    }
}
