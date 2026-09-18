//! FFV1 decoder implementation.
//!
//! Decodes FFV1 lossless video bitstreams as specified in RFC 9043.
//! Supports version 3 with range coder and CRC-32 error detection.
//! 8/10/12-bit depth, parallel multi-slice decode via rayon.

use rayon::prelude::*;

use crate::error::{CodecError, CodecResult};
use crate::frame::{FrameType, Plane, VideoFrame};
use crate::traits::VideoDecoder;
use oximedia_core::{CodecId, PixelFormat, Rational, Timestamp};

use super::crc32::crc32_mpeg2;
use super::prediction::predict_median;
use super::range_coder::SimpleRangeDecoder;
use super::types::{
    Ffv1ChromaType, Ffv1Colorspace, Ffv1Config, Ffv1Version, SliceHeader, CONTEXT_COUNT,
    INITIAL_STATE,
};

/// Decode all planes in a slice using a single shared range coder.
///
/// Per RFC 9043 §3.8.2.2.1, each slice has a single arithmetic coder stream
/// with per-plane context state arrays. All planes are decoded sequentially
/// from the same bitstream, each using their own context state.
///
/// `plane_headers` gives the (width, height) region for each plane.
/// `plane_states` is a Vec of per-plane context state arrays (each len CONTEXT_COUNT).
///
/// Returns decoded samples as `Vec<plane: Vec<row: Vec<i32>>>`.
fn decode_all_planes_in_slice(
    data: &[u8],
    plane_headers: &[SliceHeader],
    plane_states: &mut Vec<Vec<u8>>,
) -> CodecResult<Vec<Vec<Vec<i32>>>> {
    let plane_count = plane_headers.len();

    if data.len() < 2 {
        // Not enough data for range coder init; return black (zero) planes.
        let mut planes = Vec::with_capacity(plane_count);
        for header in plane_headers {
            let w = header.slice_width as usize;
            let h = header.slice_height as usize;
            let mut lines = Vec::with_capacity(h);
            for _ in 0..h {
                lines.push(vec![0i32; w]);
            }
            planes.push(lines);
        }
        return Ok(planes);
    }

    let mut decoder = SimpleRangeDecoder::new(data)?;
    let mut planes_out: Vec<Vec<Vec<i32>>> = Vec::with_capacity(plane_count);

    for (plane_idx, header) in plane_headers.iter().enumerate() {
        let w = header.slice_width as usize;
        let h = header.slice_height as usize;

        let states = plane_states
            .get_mut(plane_idx)
            .ok_or_else(|| CodecError::Internal("invalid plane index".to_string()))?;

        if w == 0 || h == 0 {
            planes_out.push(Vec::new());
            continue;
        }

        let mut lines: Vec<Vec<i32>> = Vec::with_capacity(h);
        let mut prev_line = vec![0i32; w];

        for _y in 0..h {
            let mut line = Vec::with_capacity(w);
            for x in 0..w {
                let residual = decoder.get_symbol(states)?;
                let left = if x > 0 { line[x - 1] } else { 0 };
                let top = prev_line[x];
                let top_left = if x > 0 { prev_line[x - 1] } else { 0 };
                let pred = predict_median(left, top, top_left);
                // Use saturating_add to avoid debug-mode overflow panic; correct
                // encoders produce residuals well within i32 range.
                line.push(pred.saturating_add(residual));
            }
            prev_line.clone_from(&line);
            lines.push(line);
        }
        planes_out.push(lines);
    }

    Ok(planes_out)
}

/// Map (colorspace, chroma, bits) → PixelFormat.
fn pixel_format_for_config(config: &Ffv1Config) -> PixelFormat {
    match (
        config.colorspace,
        config.chroma_type,
        config.bits_per_raw_sample,
    ) {
        (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma420, 8) => PixelFormat::Yuv420p,
        (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma420, 10) => PixelFormat::Yuv420p10le,
        (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma420, 12) => PixelFormat::Yuv420p12le,
        (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma422, 8) => PixelFormat::Yuv422p,
        (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma422, 10) => PixelFormat::Yuv422p10le,
        (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma422, 12) => PixelFormat::Yuv422p12le,
        (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma444, 8) => PixelFormat::Yuv444p,
        (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma444, 10) => PixelFormat::Yuv444p10le,
        (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma444, 12) => PixelFormat::Yuv444p12le,
        (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma420, 16) => PixelFormat::Yuv420p16le,
        (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma422, 16) => PixelFormat::Yuv422p16le,
        (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma444, 16) => PixelFormat::Yuv444p16le,
        _ => PixelFormat::Yuv420p, // safe fallback
    }
}

/// FFV1 decoder.
///
/// Implements the `VideoDecoder` trait for decoding FFV1 lossless video.
/// Supports 8/10/12-bit depths and parallel multi-slice decode via rayon.
///
/// # Usage
///
/// ```ignore
/// use oximedia_codec::ffv1::Ffv1Decoder;
/// use oximedia_codec::VideoDecoder;
///
/// let mut decoder = Ffv1Decoder::new();
/// decoder.send_packet(&compressed_data, pts)?;
/// if let Some(frame) = decoder.receive_frame()? {
///     // Process decoded frame
/// }
/// ```
pub struct Ffv1Decoder {
    /// Codec configuration (parsed from extradata or first frame).
    config: Option<Ffv1Config>,
    /// Output frame queue.
    output_queue: Vec<VideoFrame>,
    /// Whether the decoder is in flush mode.
    flushing: bool,
    /// Number of decoded frames.
    frame_count: u64,
    /// Per-plane context states for range coder (reset each keyframe).
    plane_states: Vec<Vec<u8>>,
}

impl Ffv1Decoder {
    /// Create a new FFV1 decoder.
    pub fn new() -> Self {
        Self {
            config: None,
            output_queue: Vec::new(),
            flushing: false,
            frame_count: 0,
            plane_states: Vec::new(),
        }
    }

    /// Create a decoder initialized with extradata (configuration record).
    pub fn with_extradata(extradata: &[u8]) -> CodecResult<Self> {
        let mut dec = Self::new();
        dec.parse_config(extradata)?;
        Ok(dec)
    }

    /// Parse the FFV1 configuration record from extradata.
    ///
    /// For FFV1 v3, the configuration record is a range-coded bitstream
    /// containing codec parameters. For simplicity, we also support a
    /// compact binary format used within our own container.
    fn parse_config(&mut self, data: &[u8]) -> CodecResult<()> {
        // Minimal configuration record: at least 16 bytes for our binary format.
        // Format: [version(1), colorspace(1), chroma_h_shift(1), chroma_v_shift(1),
        //          bits(1), ec(1), num_h_slices(1), num_v_slices(1),
        //          width(4 LE), height(4 LE)]  = 16 bytes minimum
        if data.len() < 16 {
            return Err(CodecError::InvalidBitstream(format!(
                "FFV1 config too short: {} bytes, need at least 16",
                data.len()
            )));
        }

        let version = Ffv1Version::from_u8(data[0])?;
        let colorspace = Ffv1Colorspace::from_u8(data[1])?;
        let h_shift = u32::from(data[2]);
        let v_shift = u32::from(data[3]);
        let chroma_type = Ffv1ChromaType::from_shifts(h_shift, v_shift)?;
        let bits_per_raw_sample = data[4];
        let ec = data[5] != 0;
        let num_h_slices = u32::from(data[6]);
        let num_v_slices = u32::from(data[7]);

        // Read width and height as little-endian u32
        let width_bytes: [u8; 4] = data[8..12]
            .try_into()
            .map_err(|_| CodecError::InvalidBitstream("bad width bytes".to_string()))?;
        let height_bytes: [u8; 4] = data[12..16]
            .try_into()
            .map_err(|_| CodecError::InvalidBitstream("bad height bytes".to_string()))?;

        let width = u32::from_le_bytes(width_bytes);
        let height = u32::from_le_bytes(height_bytes);

        let config = Ffv1Config {
            version,
            width,
            height,
            colorspace,
            chroma_type,
            bits_per_raw_sample,
            num_h_slices,
            num_v_slices,
            ec,
            range_coder_mode: version.uses_range_coder(),
            state_transition_delta: Vec::new(),
        };
        config.validate()?;

        self.init_states(&config);
        self.config = Some(config);
        Ok(())
    }

    /// Initialize per-plane context states.
    fn init_states(&mut self, config: &Ffv1Config) {
        let plane_count = config.plane_count();
        self.plane_states.clear();
        for _ in 0..plane_count {
            self.plane_states.push(vec![INITIAL_STATE; CONTEXT_COUNT]);
        }
    }

    /// Reset all context states (done at keyframes).
    fn reset_states(&mut self) {
        for states in &mut self.plane_states {
            for s in states.iter_mut() {
                *s = INITIAL_STATE;
            }
        }
    }

    /// Decode a complete frame from the given packet data.
    fn decode_frame(&mut self, data: &[u8], pts: i64) -> CodecResult<VideoFrame> {
        // Extract all needed config values upfront to avoid borrow conflicts.
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| CodecError::DecoderError("FFV1 decoder not configured".to_string()))?;

        let width = config.width;
        let height = config.height;
        let plane_count = config.plane_count();
        let ec = config.ec;
        let num_slices = config.num_slices();
        let num_h_slices = config.num_h_slices;
        let num_v_slices = config.num_v_slices;
        let max_val = config.max_sample_value();
        let bps = config.bits_per_raw_sample;
        let bytes_per_sample = config.bytes_per_sample();
        let pixel_format = pixel_format_for_config(config);

        let plane_dims: Vec<(u32, u32)> = (0..plane_count)
            .map(|i| config.plane_dimensions(i))
            .collect();

        // Release the immutable borrow on self.config before mutable operations.
        let _ = config;

        let is_keyframe = self.frame_count == 0;

        if is_keyframe {
            self.reset_states();
        }

        let mut frame = VideoFrame::new(pixel_format, width, height);
        frame.timestamp = Timestamp::new(pts, Rational::new(1, 1000));
        frame.frame_type = if is_keyframe {
            FrameType::Key
        } else {
            FrameType::Inter
        };

        // Decode all planes and assemble output.
        let planes_data = if num_slices == 1 {
            self.decode_single_slice(data, ec, plane_count, &plane_dims)?
        } else {
            decode_multi_slice_parallel(
                data,
                ec,
                num_slices,
                num_h_slices,
                num_v_slices,
                plane_count,
                &plane_dims,
            )?
        };

        // Convert decoded i32 plane data → byte-packed VideoFrame planes
        for (plane_idx, plane_lines) in planes_data.iter().enumerate() {
            let (pw, ph) = plane_dims[plane_idx];
            // stride in *samples* (not bytes)
            let stride_samples = pw as usize;
            let mut plane_data = vec![0u8; stride_samples * ph as usize * bytes_per_sample];

            for (y, line) in plane_lines.iter().enumerate() {
                if y >= ph as usize {
                    break;
                }
                for (x, &sample) in line.iter().enumerate() {
                    if x >= pw as usize {
                        break;
                    }
                    let s = sample.clamp(0, max_val);
                    let out_idx = y * stride_samples + x;
                    if bps <= 8 {
                        plane_data[out_idx] = s as u8;
                    } else {
                        plane_data[out_idx * 2] = (s & 0xFF) as u8;
                        plane_data[out_idx * 2 + 1] = ((s >> 8) & 0xFF) as u8;
                    }
                }
            }

            // stride passed to Plane is in bytes
            let stride_bytes = stride_samples * bytes_per_sample;
            frame
                .planes
                .push(Plane::with_dimensions(plane_data, stride_bytes, pw, ph));
        }

        self.frame_count += 1;
        Ok(frame)
    }

    /// Decode all planes from a single-slice packet.
    ///
    /// Per RFC 9043, all planes in a slice share a single arithmetic coder
    /// stream; they are decoded sequentially from the same bitstream.
    fn decode_single_slice(
        &mut self,
        data: &[u8],
        ec: bool,
        plane_count: usize,
        plane_dims: &[(u32, u32)],
    ) -> CodecResult<Vec<Vec<Vec<i32>>>> {
        let slice_data = if ec && data.len() >= 4 {
            let payload = &data[..data.len() - 4];
            let stored_crc_bytes: [u8; 4] = data[data.len() - 4..]
                .try_into()
                .map_err(|_| CodecError::InvalidBitstream("bad CRC bytes".to_string()))?;
            let stored_crc = u32::from_le_bytes(stored_crc_bytes);
            let computed_crc = crc32_mpeg2(payload);
            if stored_crc != computed_crc {
                return Err(CodecError::InvalidBitstream(format!(
                    "FFV1 slice CRC mismatch: stored={stored_crc:#010X}, computed={computed_crc:#010X}"
                )));
            }
            payload
        } else {
            data
        };

        // Build per-plane slice headers (full frame = single slice)
        let headers: Vec<SliceHeader> = plane_dims
            .iter()
            .map(|&(pw, ph)| SliceHeader {
                slice_x: 0,
                slice_y: 0,
                slice_width: pw,
                slice_height: ph,
            })
            .collect();

        decode_all_planes_in_slice(slice_data, &headers, &mut self.plane_states)
    }
}

/// Decode multi-slice frame data in parallel via rayon.
///
/// Each slice resets its context states independently (RFC 9043 §3.8.2.2.1),
/// so slices are data-independent and safe to parallelize.
///
/// Returns Vec<plane_lines> where `plane_lines[plane_idx][row]` holds sample data.
fn decode_multi_slice_parallel(
    data: &[u8],
    ec: bool,
    num_slices: u32,
    num_h_slices: u32,
    num_v_slices: u32,
    plane_count: usize,
    plane_dims: &[(u32, u32)],
) -> CodecResult<Vec<Vec<Vec<i32>>>> {
    let slice_data_len = data.len() / (num_slices as usize);

    // Build slice descriptors: (sy, sx, slice_idx, data_start, data_end)
    let slice_descs: Vec<(u32, u32, usize, usize, usize)> = (0..num_v_slices)
        .flat_map(|sy| {
            (0..num_h_slices).map(move |sx| {
                let slice_idx = (sy * num_h_slices + sx) as usize;
                let start = slice_idx * slice_data_len;
                let end = if slice_idx + 1 == num_slices as usize {
                    data.len()
                } else {
                    start + slice_data_len
                };
                (sy, sx, slice_idx, start, end)
            })
        })
        .collect();

    // Parallel decode: each closure decodes all planes for its slice.
    let slice_results: Vec<Result<(usize, usize, Vec<Vec<Vec<i32>>>), CodecError>> = slice_descs
        .par_iter()
        .map(|&(sy, sx, _slice_idx, start, end)| {
            // Determine actual slice segment, stripping optional trailing CRC.
            let raw_segment = &data[start..end];
            let slice_segment = if ec && raw_segment.len() >= 4 {
                let payload = &raw_segment[..raw_segment.len() - 4];
                let stored_bytes: [u8; 4] = raw_segment[raw_segment.len() - 4..]
                    .try_into()
                    .map_err(|_| {
                        CodecError::InvalidBitstream("bad CRC bytes in slice".to_string())
                    })?;
                let stored_crc = u32::from_le_bytes(stored_bytes);
                let computed_crc = crc32_mpeg2(payload);
                if stored_crc != computed_crc {
                    return Err(CodecError::InvalidBitstream(format!(
                        "FFV1 multi-slice CRC mismatch: stored={stored_crc:#010X}, computed={computed_crc:#010X}"
                    )));
                }
                payload
            } else {
                raw_segment
            };

            // Per RFC 9043: context state resets per slice; fresh state per plane.
            let mut local_plane_states: Vec<Vec<u8>> = (0..plane_count)
                .map(|_| vec![INITIAL_STATE; CONTEXT_COUNT])
                .collect();

            // Build per-plane headers for this slice
            let headers: Vec<SliceHeader> = plane_dims
                .iter()
                .enumerate()
                .map(|(plane_idx, &(pw, ph))| {
                    let slice_pw = pw / num_h_slices;
                    let slice_ph = ph / num_v_slices;
                    let actual_sw = if sx == num_h_slices - 1 {
                        pw - sx * slice_pw
                    } else {
                        slice_pw
                    };
                    let actual_sh = if sy == num_v_slices - 1 {
                        ph - sy * slice_ph
                    } else {
                        slice_ph
                    };
                    let _ = plane_idx; // plane_idx not needed beyond dims
                    SliceHeader {
                        slice_x: sx * slice_pw,
                        slice_y: sy * slice_ph,
                        slice_width: actual_sw,
                        slice_height: actual_sh,
                    }
                })
                .collect();

            let per_plane_rows = decode_all_planes_in_slice(
                slice_segment,
                &headers,
                &mut local_plane_states,
            )?;

            Ok((sy as usize, sx as usize, per_plane_rows))
        })
        .collect();

    // Propagate errors
    let decoded_slices: Vec<(usize, usize, Vec<Vec<Vec<i32>>>)> = slice_results
        .into_iter()
        .collect::<Result<Vec<_>, CodecError>>(
    )?;

    // Build a 2D lookup: (sy, sx) → per_plane_rows
    let mut grid: std::collections::HashMap<(usize, usize), Vec<Vec<Vec<i32>>>> =
        std::collections::HashMap::new();
    let mut ordered = decoded_slices;
    for (sy, sx, rows) in ordered.drain(..) {
        grid.insert((sy, sx), rows);
    }

    // Reassemble per-plane output rows from slice results.
    let mut planes_data: Vec<Vec<Vec<i32>>> = (0..plane_count).map(|_| Vec::new()).collect();

    for sy in 0..num_v_slices as usize {
        for plane_idx in 0..plane_count {
            let mut plane_band: Vec<Vec<i32>> = Vec::new();

            let (pw, ph) = plane_dims[plane_idx];
            let slice_ph = ph as usize / num_v_slices as usize;
            let actual_sh = if sy == num_v_slices as usize - 1 {
                ph as usize - sy * slice_ph
            } else {
                slice_ph
            };

            for row_in_band in 0..actual_sh {
                // Concatenate columns for this row.
                let mut full_row: Vec<i32> = Vec::with_capacity(pw as usize);
                for sx in 0..num_h_slices as usize {
                    let slice_pw = pw as usize / num_h_slices as usize;
                    let actual_sw = if sx == num_h_slices as usize - 1 {
                        pw as usize - sx * slice_pw
                    } else {
                        slice_pw
                    };
                    let slice_rows = grid.get(&(sy, sx)).ok_or_else(|| {
                        CodecError::Internal(format!("missing slice ({sy}, {sx})"))
                    })?;
                    let slice_plane_rows = slice_rows.get(plane_idx).ok_or_else(|| {
                        CodecError::Internal(format!(
                            "missing plane {plane_idx} in slice ({sy}, {sx})"
                        ))
                    })?;
                    let row = slice_plane_rows.get(row_in_band).ok_or_else(|| {
                        CodecError::Internal(format!(
                            "missing row {row_in_band} in plane {plane_idx} slice ({sy}, {sx})"
                        ))
                    })?;
                    // Trim to actual_sw in case the slice returned more
                    let take = actual_sw.min(row.len());
                    full_row.extend_from_slice(&row[..take]);
                }
                plane_band.push(full_row);
            }
            planes_data[plane_idx].extend(plane_band);
        }
    }

    Ok(planes_data)
}

impl VideoDecoder for Ffv1Decoder {
    fn codec(&self) -> CodecId {
        CodecId::Ffv1
    }

    fn send_packet(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        if self.flushing {
            return Err(CodecError::DecoderError(
                "decoder is flushing, cannot accept new packets".to_string(),
            ));
        }

        if self.config.is_none() {
            return Err(CodecError::DecoderError(
                "FFV1 decoder not configured: call with_extradata() first".to_string(),
            ));
        }

        let frame = self.decode_frame(data, pts)?;
        self.output_queue.push(frame);
        Ok(())
    }

    fn receive_frame(&mut self) -> CodecResult<Option<VideoFrame>> {
        if self.output_queue.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.output_queue.remove(0)))
        }
    }

    fn flush(&mut self) -> CodecResult<()> {
        self.flushing = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.output_queue.clear();
        self.flushing = false;
        self.frame_count = 0;
        self.reset_states();
    }

    fn output_format(&self) -> Option<PixelFormat> {
        self.config.as_ref().map(pixel_format_for_config)
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        self.config.as_ref().map(|c| (c.width, c.height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::VideoDecoder;

    fn make_config_bytes(width: u32, height: u32, bits: u8, h_shift: u8, v_shift: u8) -> Vec<u8> {
        let mut data = Vec::new();
        data.push(3); // version = V3
        data.push(0); // colorspace = YCbCr
        data.push(h_shift);
        data.push(v_shift);
        data.push(bits);
        data.push(1); // ec = true
        data.push(1); // num_h_slices
        data.push(1); // num_v_slices
        data.extend_from_slice(&width.to_le_bytes());
        data.extend_from_slice(&height.to_le_bytes());
        data
    }

    fn make_config_bytes_420_8(width: u32, height: u32) -> Vec<u8> {
        make_config_bytes(width, height, 8, 1, 1)
    }

    #[test]
    fn test_decoder_creation() {
        let dec = Ffv1Decoder::new();
        assert!(dec.config.is_none());
        assert_eq!(dec.codec(), CodecId::Ffv1);
    }

    #[test]
    fn test_decoder_with_extradata() {
        let config_data = make_config_bytes_420_8(320, 240);
        let dec = Ffv1Decoder::with_extradata(&config_data).expect("valid config");
        assert!(dec.config.is_some());
        assert_eq!(dec.dimensions(), Some((320, 240)));
        assert_eq!(dec.output_format(), Some(PixelFormat::Yuv420p));
    }

    #[test]
    fn test_decoder_invalid_config() {
        // Too short
        assert!(Ffv1Decoder::with_extradata(&[0; 5]).is_err());
        // Invalid version
        let mut bad = make_config_bytes_420_8(320, 240);
        bad[0] = 99;
        assert!(Ffv1Decoder::with_extradata(&bad).is_err());
    }

    #[test]
    fn test_decoder_not_configured() {
        let mut dec = Ffv1Decoder::new();
        assert!(dec.send_packet(&[0; 100], 0).is_err());
    }

    #[test]
    fn test_decoder_reset() {
        let config_data = make_config_bytes_420_8(16, 16);
        let mut dec = Ffv1Decoder::with_extradata(&config_data).expect("valid");
        dec.frame_count = 10;
        dec.flushing = true;
        dec.reset();
        assert_eq!(dec.frame_count, 0);
        assert!(!dec.flushing);
    }

    #[test]
    fn test_decoder_flush() {
        let config_data = make_config_bytes_420_8(16, 16);
        let mut dec = Ffv1Decoder::with_extradata(&config_data).expect("valid");
        dec.flush().expect("flush ok");
        assert!(dec.flushing);
        // Should reject new packets after flush
        assert!(dec.send_packet(&[0; 100], 0).is_err());
    }

    #[test]
    fn test_pixel_format_dispatch_10bit() {
        let config = Ffv1Config {
            width: 16,
            height: 16,
            bits_per_raw_sample: 10,
            chroma_type: Ffv1ChromaType::Chroma420,
            colorspace: Ffv1Colorspace::YCbCr,
            ..Default::default()
        };
        let dec = Ffv1Decoder {
            config: Some(config),
            output_queue: Vec::new(),
            flushing: false,
            frame_count: 0,
            plane_states: Vec::new(),
        };
        assert_eq!(dec.output_format(), Some(PixelFormat::Yuv420p10le));
    }

    #[test]
    fn test_pixel_format_dispatch_12bit_444() {
        let config = Ffv1Config {
            width: 16,
            height: 16,
            bits_per_raw_sample: 12,
            chroma_type: Ffv1ChromaType::Chroma444,
            colorspace: Ffv1Colorspace::YCbCr,
            ..Default::default()
        };
        let dec = Ffv1Decoder {
            config: Some(config),
            output_queue: Vec::new(),
            flushing: false,
            frame_count: 0,
            plane_states: Vec::new(),
        };
        assert_eq!(dec.output_format(), Some(PixelFormat::Yuv444p12le));
    }
}
